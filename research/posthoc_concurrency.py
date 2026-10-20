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
import subprocess
import sys
import threading
import time

_context = threading.local()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.'+str(os.getpid())+'.'+str(threading.get_ident())+'.tmp')
    write_once(temporary, value); temporary.replace(path)


def event(policy, name, **fields):
    path = Path(policy['admission_root'])/'events.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
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


def foreign_counts(policy, proc_root=Path('/proc')):
    """CUDA-visible live processes outside the two leased cohorts.

    Existing C/F/GAC controllers retain their single external slot. Unknown
    CUDA-visible processes consume capacity too. CPU observers are excluded.
    """
    managed = tuple(str(Path(c['control']))+'/' for c in policy['cohorts'].values())
    counts = [0]*4
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
            for token in visible.split(','):
                if token.strip() in ('0', '1', '2', '3'):
                    counts[int(token.strip())] += 1
        except (OSError, ValueError):
            continue
    return counts


def resource_probe(policy, gpu):
    response = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free',
        '--format=csv,noheader,nounits'], text=True)
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
    root = Path(policy['admission_root']); root.mkdir(parents=True, exist_ok=True)
    lease = None
    while lease is None:
        if getattr(_context, 'stop', None) is not None and _context.stop.is_set():
            raise RuntimeError('Unlaunched admission cancelled after infrastructure incident')
        with (root/f'gpu-{gpu}.lock').open('a') as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            state = root/f'gpu-{gpu}-last-start.json'
            last = read(state)['time'] if state.exists() else float('-inf')
            external = max(policy['reserved_external_slots'], foreign(policy)[gpu])
            if now()-last >= policy['stagger_seconds'] and probe(policy, gpu):
                for index in range(external, policy['gpu_tasks']):
                    candidate = (root/f'gpu-{gpu}-slot-{index}.lock').open('a')
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


def install(policy, kind, module, spec):
    """Apply explicit runtime overrides in memory; imported files remain immutable."""
    original_check, original_command = module.check, module.command
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
