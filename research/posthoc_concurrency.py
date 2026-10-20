"""Operational adapter for immutable posthoc controllers; no learner changes.

Run this file directly. Each original release stays importable at its original
path. Only admission, concurrency, the Q cohort's resource-only predecessor,
and the CPU observer's GL backend change under a separately hashed policy.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

_context = threading.local()
_event_mutex = threading.Lock()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.'+str(os.getpid())+'.'+str(threading.get_ident())+'.tmp')
    write_once(temporary, value); temporary.replace(path)


def event(policy, name, **fields):
    filename = f'events-{os.getpid()}.jsonl' if policy.get('single_writer_event_journal') else 'events.jsonl'
    path = Path(policy['admission_root'])/filename
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    with _event_mutex:
        with path.open('a') as stream:
            if not policy.get('single_writer_event_journal'):
                fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(dict(time=time.time(), event=name, **fields), allow_nan=False)+'\n')


def load_policy(path, expected):
    path = Path(path).resolve()
    if digest(path) != expected:
        raise ValueError('Operational policy hash differs')
    policy = read(path)
    if (policy['format'] != 'posthoc-concurrency-v1'
            or policy['job_id'] != 'a21566001929515008267857'
            or policy['gpu_tasks'] != 8 or policy['stagger_seconds'] != 30
            or policy['reserved_external_slots'] != 1 or set(policy['cohorts']) != {'Q', 'M'}
            or digest(__file__) != policy['adapter_sha256']):
        raise ValueError('Unregistered operational adapter or resource policy')
    policy.update(_path=str(path), _sha=expected)
    return policy


def receipt_job_identity(process, proc_root):
    """Find an index-addressed receipt runner through its shell/child chain."""
    for _ in range(8):
        try:
            arguments = [x.decode(errors='replace') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            if len(arguments) >= 3 and arguments[1] == '-c':
                arguments = shlex.split(arguments[2])
            for index, value in enumerate(arguments):
                if (value.startswith('/workspace/releases/')
                        and Path(value).name in ('posthoc_runner.py', 'fql_runner.py', 'base_runtime.py')
                        and arguments[index+1:index+2] == ['run'] and len(arguments) > index+2):
                    return ('receipt-job', value, int(arguments[index+2]))
            parent = next(row.split(':', 1)[1].strip() for row in (process/'status').read_text().splitlines() if row.startswith('PPid:'))
            process = proc_root/parent
        except (OSError, ValueError, StopIteration):
            return None
    return None


def foreign_counts(policy, proc_root=Path('/proc')):
    """CUDA-visible live processes outside the two leased cohorts.

    Existing C/F/GAC controllers retain their single external slot. Unknown
    CUDA-visible processes consume capacity too. CPU observers are excluded.
    """
    managed = tuple(str(Path(c['control']))+'/' for c in policy['cohorts'].values())
    workloads = [set() for _ in range(4)]
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process/'cmdline').read_bytes().split(b'\0')
            env = dict(x.split(b'=', 1) for x in (process/'environ').read_bytes().split(b'\0') if b'=' in x)
            visible = env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()
            if not visible or env.get(b'JAX_PLATFORMS') == b'cpu':
                continue
            if b'research.actor_q_entry' in argv and any(
                    a.decode(errors='replace').startswith(managed) for a in argv):
                continue
            # A leased engineering GPU validator is part of the same lease.
            if b'validate' in argv and any(a.decode(errors='replace').startswith(managed) for a in argv):
                continue
            arguments = [a.decode(errors='replace') for a in argv]
            if len(arguments) >= 3 and arguments[1] == '-c':
                arguments = shlex.split(arguments[2])
            flags = dict(a[2:].split('=', 1) for a in arguments if a.startswith('--') and '=' in a)
            # The original reference scheduler uses shell -> receipt wrapper ->
            # training child. These are one GPU job, not three GPU contexts.
            receipt_identity = receipt_job_identity(process, proc_root)
            if receipt_identity is not None:
                identity = receipt_identity
            elif (any(a.startswith('/workspace/releases/posthoc-reference-') for a in arguments)
                    and 'seed' in flags and 'run_group' in flags):
                identity = ('reference', flags['run_group'], flags['seed'])
            else:
                identity = ('pid', process.name)
            for token in visible.split(','):
                if token.strip() in ('0', '1', '2', '3'):
                    workloads[int(token.strip())].add(identity)
        except (OSError, ValueError):
            continue
    return [len(group) for group in workloads]


def slot_indices(policy, external, identity):
    end = policy['gpu_tasks']
    # Q r1 is already observing real children. Leave its occupied/pending
    # slots 3..7 untouched; M can fill slots 1..2 without taking those away.
    if identity.startswith('M:') and policy.get('map_fill_only_unused_slots_until_q_terminal'):
        control = Path(policy['cohorts']['Q']['control'])
        if not (control/'continuation/terminal.json').exists():
            end = 3
    return range(external, end)


def resource_probe(policy, gpu):
    try:
        response = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False
    rows = [[part.strip() for part in line.split(',')] for line in response.splitlines()]
    if len(rows) != 4 or any('4090' not in row[1] for row in rows):
        raise ValueError('Expected existing four 4090 devices')
    if float(next(row[2] for row in rows if row[0] == str(gpu))) < policy['minimum_free_gpu_mib']:
        return False
    group = Path('/sys/fs/cgroup')
    if (group/'memory.max').exists():
        maximum = (group/'memory.max').read_text().strip()
        if maximum != 'max' and int((group/'memory.current').read_text()) > int(maximum)*.85:
            return False
    return True


@contextmanager
def admission(policy, gpu, identity, *, probe=resource_probe, foreign=foreign_counts, sleep=time.sleep, now=time.time):
    """Cross-process lock leases; cooldown and reservations share a GPU mutex."""
    root = Path(policy['admission_root'])
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    # Append-open can wait on a remote filesystem's write ownership before
    # flock is reached. Precreated, zero-byte files need no data writes.
    lock_mode = 'rb' if policy.get('read_only_precreated_lockfiles') else 'a'
    mutex_root = Path(policy.get('admission_mutex_root', root))
    mutex_root.mkdir(parents=True, exist_ok=True)
    separate_mutex = mutex_root != root
    mutex_mode = 'a' if separate_mutex else lock_mode
    lease = None
    while lease is None:
        if getattr(_context, 'stop', None) is not None and _context.stop.is_set():
            raise RuntimeError('Unlaunched admission cancelled after infrastructure incident')
        with (mutex_root/f'gpu-{gpu}.lock').open(mutex_mode) as mutex:
            # Never enter a blocking remote flock RPC: the shared backend can
            # strand its waiter even after other controllers release the file.
            try:
                fcntl.flock(mutex, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                sleep(1)
                continue
            state = mutex_root/f'gpu-{gpu}-last-start.json'
            last = read(state)['time'] if state.exists() else float('-inf')
            # A separate M mutex never edits the live Q coordinator's files.
            # Observe its latest start too; the 30s guarantee is per cohort,
            # since the already-running Q process cannot observe M's mutex.
            shared_state = root/f'gpu-{gpu}-last-start.json'
            if separate_mutex and shared_state.exists():
                last = max(last, read(shared_state)['time'])
            external = max(policy['reserved_external_slots'], foreign(policy)[gpu])
            if now()-last >= policy['stagger_seconds'] and probe(policy, gpu):
                for index in slot_indices(policy, external, identity):
                    candidate = (root/f'gpu-{gpu}-slot-{index}.lock').open(lock_mode)
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        candidate.close(); continue
                    lease = candidate
                    atomic(state, dict(time=now(), identity=identity, pid=os.getpid()))
                    event(policy, 'admitted', gpu=gpu, slot=index, identity=identity, external_reserved=external)
                    break
        if lease is None:
            sleep(1)
    try:
        yield
    finally:
        event(policy, 'released', gpu=gpu, identity=identity)
        lease.close()


def run_concurrent(rows, operation, *, workers=8):
    if len(rows) not in (40, 60) or [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Expected unchanged ordered Q40 or M60 identities')
    queues = [deque(rows[gpu::4]) for gpu in range(4)]
    mutex = threading.Lock(); stopped = threading.Event(); output = []; errors = []
    def lane(gpu):
        _context.stop = stopped
        while not stopped.is_set():
            with mutex:
                if stopped.is_set() or not queues[gpu]:
                    return
                row = queues[gpu].popleft()
            try:
                result = operation(row, gpu)
                with mutex:
                    output.append(result)
            except Exception as error:
                with mutex:
                    errors.append(dict(gpu=gpu, incident_index=row['index'], error=str(error)))
                    stopped.set()
                return
    with ThreadPoolExecutor(max_workers=4*workers) as pool:
        list(pool.map(lane, [gpu for _ in range(workers) for gpu in range(4)]))
    return output, errors


def primary_gate(policy):
    """Technical gate only: A72/B16 complete and all eight C UTD1 exits observed."""
    evidence = {}
    for key, expected in policy['completed_predecessors'].items():
        root = Path(expected['control']); terminal = root/'terminal.json'; exit_path = root/'scheduler-exit.json'
        if not terminal.exists() or not exit_path.exists():
            return dict(stage='waiting_for_primary_technical_predecessors')
        value, observed = read(terminal), read(exit_path)
        if (value.get('source_commit') != expected['source_commit'] or value.get('status') != 'complete'
                or value.get('expected_runs') != expected['runs'] or value.get('complete_runs') != expected['runs']
                or type(observed.get('returncode')) is not int or observed['returncode'] != 0):
            raise ValueError('Primary technical predecessor differs: '+key)
        evidence[key] = dict(terminal_sha256=digest(terminal), exit_sha256=digest(exit_path))
    root = Path(policy['reference_control']); selected = {}
    for path in (root/'runs').glob('run-*.json'):
        value = read(path); identity = (value.get('name'), value.get('seed'))
        if identity not in {(f'aa_posthoc_reference_{task}_utd1', seed) for task in ('triple2', 'double2') for seed in (10, 11, 12, 13)}:
            continue
        if identity in selected:
            raise ValueError('Duplicate C UTD1 receipt')
        if value.get('status') != 'complete':
            return dict(stage='waiting_for_reference_utd1_exits')
        if value.get('source_commit') != policy['reference_source'] or type(value.get('returncode')) is not int or value['returncode'] != 0:
            raise ValueError('Reference UTD1 actual exit/source differs')
        selected[identity] = digest(path)
    if len(selected) != 8:
        return dict(stage='waiting_for_reference_utd1_exits')
    return dict(stage='upstream_terminal', scope='operational_priority_amendment_not_whole_C16_terminal',
        predecessor_evidence=evidence, reference_utd1_receipts=[dict(name=k[0], seed=k[1], sha256=v) for k,v in sorted(selected.items())],
        operational_policy_sha256=policy['_sha'], performance_threshold=None)


def map_gate(policy, spec, original_gate):
    evidence = original_gate(spec)  # Retain bound Q source/spec/owner integrity.
    if evidence['stage'] == 'upstream_terminal':
        return evidence
    root = Path(policy['cohorts']['Q']['control'])
    ready, launch = root/'engineering-gpu-r1/ready.json', root/'science/launch.json'
    if not ready.exists() or not launch.exists():
        return dict(stage='waiting_for_actor_q_engineering_and_admission')
    eng, science = read(ready), read(launch)
    if (eng.get('status') != 'complete' or len(eng.get('records', [])) != 4
            or any(row.get('status') != 'complete' for row in eng['records'])
            or science.get('plan_sha256') != read(policy['cohorts']['Q']['spec'])['plan_sha256']
            or science.get('spec_sha256') != policy['cohorts']['Q']['spec_sha256']
            or science.get('engineering_ready_sha256') != digest(ready)):
        raise ValueError('Q engineering/science admission evidence differs')
    for path in (root/'science/runs').glob('*.json'):
        if read(path).get('status') in ('failed', 'needs_review'):
            return dict(stage='waiting_for_primary_technical_incident_review')
    return dict(stage='upstream_terminal', scope='shared_independent_cohort_admission_not_Q_science_terminal',
        engineering_ready_sha256=digest(ready), science_launch_sha256=digest(launch),
        operational_policy_sha256=policy['_sha'], performance_threshold=None)


def supervise_map(module, spec):
    """Same six engineering identities/validators, with parallel admission.

    Scientific argv and collect are the original bound controller's. Only the
    serial engineering loop and resource predecessor are operational changes.
    """
    plan = module.check(spec); evidence = module.gate(spec)
    if evidence['stage'] != 'upstream_terminal' or not module.idle_gpus():
        raise RuntimeError('Technical/shared-admission prerequisite changed')
    control = Path(spec['control']); data = module.verify_data(spec)
    eng = control/'engineering-gpu-r1'
    if eng.exists():
        if not spec.get('_resume_engineering') or read(eng/'launch.json')['spec_sha256'] != spec['_spec_sha256']:
            raise FileExistsError('Existing engineering needs explicit bound read-only reconciliation')
        module.write_once(eng/('operational-reconciliation-'+spec['_operational_sha'][:12]+'.json'),
            dict(spec_sha256=spec['_spec_sha256'], gate=evidence, data=data, no_training_retries=True))
    else:
        eng.mkdir(exist_ok=False)
        module.write_once(eng/'launch.json', dict(spec_sha256=spec['_spec_sha256'], gate=evidence, data=data))
    cases = module.engineering_cases(plan)
    with ThreadPoolExecutor(max_workers=6) as pool:
        engineering = list(pool.map(lambda row: module.execute_case(spec, row, eng, row['index'] % 4, engineering=True), cases))
    if any(row['status'] != 'complete' for row in engineering):
        raise RuntimeError('4090 engineering failed; preserve attempts, no science launch')
    module.technical.validate_groups([dict(row, driver_returncode=row['returncode']) for row in engineering])
    if module.verify_data(spec) != data or not module.idle_gpus():
        raise ValueError('Data/shared admission changed before science')
    module.write_once(eng/'ready.json', dict(status='complete', spec_sha256=spec['_spec_sha256'],
        cloud_4090_validation=True, tested_gpu_indices=sorted({row['gpu'] for row in engineering}), records=engineering))
    science = control/'science'; science.mkdir(exist_ok=False)
    module.write_once(science/'launch.json', dict(spec_sha256=spec['_spec_sha256'], plan_sha256=module.study.PLAN_SHA,
        engineering_ready_sha256=digest(eng/'ready.json'), gate=evidence, data=data, started=time.time()))
    _, errors = module.run_lanes(plan['cases'], lambda row, gpu: module.execute_case(spec, row, science, gpu))
    records = []
    for row in plan['cases']:
        path = science/'runs'/f"case-{row['index']:03d}.json"
        records.append(read(path) if path.exists() else dict(index=row['index'], task=row['task'], method=row['method'],
            execution_map=row['execution_map'], seed=row['seed'], source_commit=module.common.SOURCE,
            status='not_started', returncode=None, driver_exit_observed=False))
    module.check(spec)
    if module.verify_data(spec) != data:
        raise ValueError('Dataset identity/stat changed during science')
    result = module.study.collect(plan, records)
    result.update(plan_sha256=module.study.PLAN_SHA, spec_sha256=spec['_spec_sha256'], lane_errors=errors)
    module.write_once(Path(spec['reports'])/'matched-map-report.json', result)
    return 0 if result['status'] == 'complete' and not errors else 2


def install(policy, kind, module, spec):
    """Apply explicit runtime overrides in memory; imported files remain immutable."""
    original_check, original_command, original_gate = module.check, module.command, module.gate
    original_environment, original_execute = module.environment, module.execute_case
    original_write = module.write_once
    def check(value):
        if digest(policy['_path']) != policy['_sha'] or digest(__file__) != policy['adapter_sha256']:
            raise ValueError('Operational adapter changed during execution')
        return original_check(value)
    def command(value, mode, *extra):
        if mode in ('own', 'supervise'):
            return [value['python'], str(Path(__file__).resolve()), mode, '--policy', policy['_path'],
                '--policy-sha', policy['_sha'], '--kind', kind, '--spec', value['_spec_path'], *map(str, extra)]
        return original_command(value, mode, *extra)
    def environment(value, gpu=None):
        result = original_environment(value, gpu)
        # EGL is available in the existing image; OSMesa is not. No rendering,
        # learner flags, package versions or training RNG are changed.
        result['MUJOCO_GL'] = 'egl'
        return result
    def execute(value, row, root, gpu, *, engineering=False):
        identity = f'{kind}:{"engineering" if engineering else "science"}:{row["index"]}'
        root = Path(root); tag = f'case-{row["index"]:03d}'
        if kind == 'M' and engineering and policy.get('resume_verified_engineering') and (root/'runs'/(tag+'.json')).exists():
            record = read(root/'runs'/(tag+'.json'))
            if (record.get('status') != 'complete' or record.get('returncode') != 0
                    or record.get('validation_returncode') != 0 or read(root/'cases'/(tag+'.json')) != row):
                raise ValueError('Unknown/failed engineering attempt cannot be retried')
            for label in ('driver', 'validator'):
                path = Path(record[label+'_exit_path'])
                if (digest(path) != record[label+'_exit_sha256']
                        or type(read(path).get('returncode')) is not int or read(path)['returncode'] != 0):
                    raise ValueError('Existing engineering actual exit differs')
            output = root/'operational-cpu-rechecks'/(tag+'-'+policy['_sha'][:12]+'.json')
            argv = original_command(value, 'validate', '--case', root/'cases'/(tag+'.json'),
                '--initial', root/'initial'/(tag+'.json'), '--output', output)
            started = time.time()
            with (root/'logs'/(tag+'-recheck-'+policy['_sha'][:12]+'.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=value['_root'], env=environment(value), stdout=log, stderr=subprocess.STDOUT)
                write_once(root/'processes'/(tag+'-recheck-'+policy['_sha'][:12]+'-started.json'), dict(pid=child.pid, argv=argv, started=started))
                code = child.wait()
            write_once(root/'processes'/(tag+'-recheck-'+policy['_sha'][:12]+'-exit.json'),
                dict(pid=child.pid, returncode=code, started=started, ended=time.time(), observed_by_pid=os.getpid()))
            if code or not output.exists():
                raise RuntimeError('Existing engineering read-only reconciliation failed')
            return record
        with admission(policy, gpu, identity):
            result = original_execute(value, row, root, gpu, engineering=engineering)
            if kind == 'Q' and engineering and result['status'] == 'complete':
                tag = f'case-{row["index"]:03d}'; root = Path(root)
                output = root/'operational-cpu-restores'/(tag+'.json')
                argv = original_command(value, 'validate', '--case', root/'cases'/(tag+'.json'),
                    '--initial', root/'initial'/(tag+'.json'), '--output', output)
                started = time.time()
                with (root/'logs'/(tag+'-operational-cpu-restore.log')).open('x') as log:
                    child = subprocess.Popen(argv, cwd=value['_root'], env=environment(value), stdout=log, stderr=subprocess.STDOUT)
                    write_once(root/'processes'/(tag+'-operational-cpu-restore-started.json'), dict(pid=child.pid, argv=argv, started=started))
                    code = child.wait()
                write_once(root/'processes'/(tag+'-operational-cpu-restore-exit.json'),
                    dict(pid=child.pid, returncode=code, started=started, ended=time.time(), observed_by_pid=os.getpid()))
                if code or not output.exists():
                    raise RuntimeError('Operational independent CPU restore failed; no science retry')
            return result
    def bound_write(path, value):
        if Path(path).name in {'terminal.json', 'launch.json', 'ready.json', 'supervisor-intent.json',
                'supervisor-started.json', 'actor-q-ablation-report.json', 'matched-map-report.json'}:
            value = dict(value, operational_policy_sha256=policy['_sha'],
                operational_adapter_sha256=policy['adapter_sha256'], gpu_tasks=policy['gpu_tasks'],
                reserved_external_slots=policy['reserved_external_slots'], stagger_seconds=policy['stagger_seconds'])
        return original_write(path, value)
    module.check, module.command, module.environment = check, command, environment
    module.execute_case, module.run_lanes, module.write_once = execute, run_concurrent, bound_write
    module.idle_gpus = lambda: any(resource_probe(policy, gpu) for gpu in range(4))
    if kind == 'Q':
        module.gate = lambda _: primary_gate(policy)
    elif policy.get('map_fill_only_unused_slots_until_q_terminal'):
        spec.update(_resume_engineering=bool(policy.get('resume_verified_engineering')), _operational_sha=policy['_sha'])
        module.gate = lambda value: map_gate(policy, value, original_gate)
        module.supervise = lambda value: supervise_map(module, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'own', 'supervise'))
    parser.add_argument('--policy', type=Path, required=True); parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--kind', choices=('Q', 'M'), required=True); parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(); policy = load_policy(args.policy, args.policy_sha)
    bound = policy['cohorts'][args.kind]
    if str(args.spec.resolve()) != bound['spec'] or digest(args.spec) != bound['spec_sha256']:
        raise ValueError('Original immutable controller specification differs')
    sys.path.insert(0, str(args.spec.resolve().parent))
    module = importlib.import_module(bound['module']); spec = module.load_spec(args.spec)
    install(policy, args.kind, module, spec)
    if args.mode == 'inspect':
        print(json.dumps(module.inspect(spec))); return
    raise SystemExit(module.own(spec) if args.mode == 'own' else module.supervise(spec))


if __name__ == '__main__':
    main()
