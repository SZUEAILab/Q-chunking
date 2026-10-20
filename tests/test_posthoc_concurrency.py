from collections import Counter
import json
import multiprocessing
from pathlib import Path
import threading
import time

import pytest

from research import posthoc_concurrency as r


@pytest.mark.parametrize('size', [40, 60])
def test_each_frozen_identity_once_same_gpu_and_real_parallelism(size):
    rows = [dict(index=i) for i in range(size)]
    active = [0]*4; maximum = [0]*4; seen = []; lock = threading.Lock()
    def execute(row, gpu):
        assert gpu == row['index'] % 4
        with lock:
            active[gpu] += 1; maximum[gpu] = max(maximum[gpu], active[gpu]); seen.append(row['index'])
        time.sleep(.025)
        with lock:
            active[gpu] -= 1
        return dict(index=row['index'], status='failed' if row['index'] == 8 else 'complete')
    output, errors = r.run_concurrent(rows, execute)
    assert not errors and sorted(seen) == list(range(size))
    assert all(1 < n <= 8 for n in maximum)
    assert next(row for row in output if row['index'] == 8)['status'] == 'failed'


def test_unknown_launch_never_retried_and_unassigned_cases_remain_missing():
    seen = []
    def execute(row, gpu):
        seen.append(row['index'])
        if row['index'] == 0:
            raise OSError('unknown launch')
        return row
    output, errors = r.run_concurrent([dict(index=i) for i in range(40)], execute)
    assert seen == [0] and not output and errors[0]['incident_index'] == 0


def test_duplicate_identity_rejected_before_any_operation():
    rows = [dict(index=i) for i in range(40)]; rows[2]['index'] = 1
    with pytest.raises(ValueError):
        r.run_concurrent(rows, lambda *_: pytest.fail('must not launch'))


def _lease_worker(policy, queue):
    with r.admission(policy, 0, str(time.time_ns()), probe=lambda *_: True, foreign=lambda _: [1]*4, sleep=lambda _: time.sleep(.005)):
        queue.put(('start', time.time())); time.sleep(.12); queue.put(('end', time.time()))


@pytest.mark.parametrize('read_only', [False, True])
def test_cross_process_slot_limit_reserves_existing_work_and_staggers(tmp_path, read_only):
    policy = dict(admission_root=str(tmp_path), gpu_tasks=3, reserved_external_slots=1, stagger_seconds=.025, read_only_precreated_lockfiles=read_only)
    if read_only:
        for name in ['gpu-0.lock','gpu-0-slot-1.lock','gpu-0-slot-2.lock']:(tmp_path/name).touch()
    context = multiprocessing.get_context('fork'); queue = context.Queue()
    children = [context.Process(target=_lease_worker, args=(policy, queue)) for _ in range(5)]
    for child in children:
        child.start()
    events = [queue.get(timeout=10) for _ in range(10)]
    for child in children:
        child.join(10); assert child.exitcode == 0
    active = maximum = 0
    for kind, _ in sorted(events, key=lambda x: x[1]):
        active += 1 if kind == 'start' else -1; maximum = max(maximum, active)
    assert maximum == 2 and active == 0
    starts = sorted(t for kind, t in events if kind == 'start')
    assert all(b-a >= .02 for a,b in zip(starts, starts[1:]))
    assert Counter(json.loads(line)['event'] for line in (tmp_path/'events.jsonl').read_text().splitlines()) == {'admitted': 5, 'released': 5}


def test_foreign_processes_counted_but_leased_driver_and_cpu_validator_excluded(tmp_path):
    policy = dict(cohorts={'Q': dict(control='/work/Q'), 'M': dict(control='/work/M')})
    cases = [(['python', 'posthoc_entry.py'], {'CUDA_VISIBLE_DEVICES': '0'}),
        (['python', 'unknown.py'], {'CUDA_VISIBLE_DEVICES': '0,1'}),
        (['python', '-m', 'research.actor_q_entry', '--case', '/work/Q/science/cases/case-000.json'], {'CUDA_VISIBLE_DEVICES': '1'}),
        (['python', 'validate', '--case', '/work/M/engineering-gpu-r1/cases/case-000.json'], {'CUDA_VISIBLE_DEVICES': '2'}),
        (['python', 'other'], {'CUDA_VISIBLE_DEVICES': '3', 'JAX_PLATFORMS': 'cpu'})]
    for index, (argv, env) in enumerate(cases, 100):
        p = tmp_path/str(index); p.mkdir(); (p/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv))
        (p/'environ').write_bytes(b'\0'.join((k+'='+v).encode() for k,v in env.items()))
    assert r.foreign_counts(policy, tmp_path) == [2, 1, 0, 0]


@pytest.fixture
def gate(tmp_path):
    policy = dict(completed_predecessors={}, reference_control=str(tmp_path/'C'), reference_source='source', _sha='policy')
    for name, count in [('A', 72), ('B', 16)]:
        root = tmp_path/name; root.mkdir()
        r.write_once(root/'terminal.json', dict(status='complete', source_commit=name, expected_runs=count, complete_runs=count))
        r.write_once(root/'scheduler-exit.json', dict(returncode=0))
        policy['completed_predecessors'][name] = dict(control=str(root), source_commit=name, runs=count)
    root = tmp_path/'C/runs'; root.mkdir(parents=True)
    for index in range(8):
        r.write_once(root/f'run-{index}.json', dict(name='aa_posthoc_reference_'+('triple2' if index < 4 else 'double2')+'_utd1',
            seed=10+index%4, source_commit='source', status='complete', returncode=0))
    return policy, root


def test_primary_gate_does_not_require_utd20_or_read_outcome_direction(gate):
    policy, root = gate
    r.write_once(root/'utd20.json', dict(name='aa_posthoc_reference_double2_utd20', seed=10, status='running'))
    value = r.primary_gate(policy)
    assert value['stage'] == 'upstream_terminal' and len(value['reference_utd1_receipts']) == 8
    assert value['performance_threshold'] is None
    assert not (root.parent/'terminal.json').exists()


@pytest.mark.parametrize('failure', ['missing', 'running', 'source', 'exit', 'bool_exit', 'duplicate', 'predecessor_exit'])
def test_primary_gate_requires_all_real_matching_receipts(gate, failure):
    policy, root = gate; path = root/'run-7.json'; value = r.read(path)
    if failure == 'missing':
        path.unlink(); assert r.primary_gate(policy)['stage'].startswith('waiting'); return
    if failure == 'running':
        value['status'] = 'running'; path.write_text(json.dumps(value)); assert r.primary_gate(policy)['stage'].startswith('waiting'); return
    if failure == 'source': value['source_commit'] = 'other'
    elif failure == 'exit': value['returncode'] = 1
    elif failure == 'bool_exit': value['returncode'] = False
    elif failure == 'duplicate': r.write_once(root/'run-duplicate.json', value)
    else:
        (Path(policy['completed_predecessors']['A']['control'])/'scheduler-exit.json').write_text('{"returncode": null}')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): r.primary_gate(policy)


def test_policy_and_adapter_identity_checked(tmp_path):
    policy = dict(format='posthoc-concurrency-v1', job_id='a21566001929515008267857', gpu_tasks=8,
        stagger_seconds=30, reserved_external_slots=1, cohorts={'Q': {}, 'M': {}}, adapter_sha256=r.digest(r.__file__))
    p = tmp_path/'policy.json'; r.write_once(p, policy)
    assert r.load_policy(p, r.digest(p))['gpu_tasks'] == 8
    sha = r.digest(p); p.write_text('{}')
    with pytest.raises(ValueError): r.load_policy(p, sha)


def test_cancelled_unlaunched_admission_never_acquires_slot(tmp_path):
    stop = threading.Event(); stop.set(); r._context.stop = stop
    try:
        with pytest.raises(RuntimeError, match='cancelled'):
            with r.admission(dict(admission_root=str(tmp_path)), 0, 'unlaunched'):
                pytest.fail('must not run')
    finally:
        del r._context.stop


def test_reference_shell_receipt_wrapper_and_training_child_are_one_gpu_workload(tmp_path):
    policy = dict(cohorts={'Q': dict(control='/work/Q'), 'M': dict(control='/work/M')})
    prefix = '/workspace/releases/posthoc-reference-e20834c88c10/'
    job = ['--seed=10', '--run_group=reference_double_utd20']
    cases = [['sh', '-c', 'python '+prefix+'posthoc_entry.py '+' '.join(job)],
        ['python', prefix+'posthoc_entry.py', *job],
        ['python', prefix+'source/main_rlpd_posthoc_reference.py', *job],
        ['python', prefix+'posthoc_entry.py', '--seed=11', '--run_group=reference_double_utd20']]
    for index, argv in enumerate(cases, 100):
        p = tmp_path/str(index); p.mkdir(); (p/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv))
        (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0')
    assert r.foreign_counts(policy, tmp_path) == [2, 0, 0, 0]


def test_map_fills_only_unused_slots_before_q_terminal(tmp_path):
    policy = dict(gpu_tasks=8, map_fill_only_unused_slots_until_q_terminal=True,
        cohorts={'Q': dict(control=str(tmp_path))})
    assert list(r.slot_indices(policy, 1, 'M:science:0')) == [1, 2]
    assert list(r.slot_indices(policy, 1, 'Q:science:0')) == list(range(1, 8))
    r.write_once(tmp_path/'continuation/terminal.json', {'status': 'complete'})
    assert list(r.slot_indices(policy, 1, 'M:science:0')) == list(range(1, 8))


@pytest.fixture
def map_admission(tmp_path):
    root = tmp_path/'Q'; qspec = tmp_path/'qspec.json'; r.write_once(qspec, {'plan_sha256': 'plan'})
    policy = dict(cohorts={'Q': dict(control=str(root), spec=str(qspec), spec_sha256='spec')}, _sha='policy')
    ready = root/'engineering-gpu-r1/ready.json'
    r.write_once(ready, dict(status='complete', records=[{'status': 'complete'}]*4))
    r.write_once(root/'science/launch.json', dict(plan_sha256='plan', spec_sha256='spec', engineering_ready_sha256=r.digest(ready)))
    return policy, root


def test_map_shared_admission_needs_engineering_not_favorable_q_outcomes(map_admission):
    policy, root = map_admission
    value = r.map_gate(policy, {}, lambda _: {'stage': 'waiting_for_existing_actor_q_queue'})
    assert value['stage'] == 'upstream_terminal' and value['performance_threshold'] is None
    assert value['scope'] == 'shared_independent_cohort_admission_not_Q_science_terminal'
    assert not (root/'continuation/terminal.json').exists()
    r.write_once(root/'science/runs/case-000.json', {'status': 'failed'})
    assert r.map_gate(policy, {}, lambda _: {'stage': 'waiting'})['stage'] == 'waiting_for_primary_technical_incident_review'


def test_map_admission_rejects_missing_or_drifted_engineering_proof(map_admission):
    policy, root = map_admission; path = root/'science/launch.json'; value = r.read(path)
    value['engineering_ready_sha256'] = 'other'; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): r.map_gate(policy, {}, lambda _: {'stage': 'waiting'})
    (root/'engineering-gpu-r1/ready.json').unlink()
    assert r.map_gate(policy, {}, lambda _: {'stage': 'waiting'})['stage'].startswith('waiting')


@pytest.mark.parametrize('engineering_failure', [False, True])
def test_parallel_map_engineering_is_once_and_failure_cannot_start_science(tmp_path, engineering_failure):
    from types import SimpleNamespace
    plan = {'cases': [dict(index=i, task='task', method='full', execution_map='map', seed=i) for i in range(60)]}
    engineering = [dict(index=i, original_argv=['frozen', str(i)]) for i in range(6)]
    seen = []; lock = threading.Lock()
    def execute(spec, row, root, gpu, *, engineering=False):
        with lock: seen.append((engineering, row['index'], tuple(row.get('original_argv', []))))
        value = dict(row, gpu=gpu, returncode=0, status='failed' if engineering and engineering_failure and row['index']==0 else 'complete')
        r.write_once(Path(root)/'runs'/f"case-{row['index']:03d}.json", value)
        return value
    spec = dict(control=str(tmp_path/'control'), reports=str(tmp_path/'reports'), _spec_sha256='spec')
    Path(spec['control']).mkdir()  # Created by the real owner before supervise.
    module = SimpleNamespace(check=lambda _: plan, gate=lambda _: {'stage': 'upstream_terminal'}, idle_gpus=lambda: True,
        verify_data=lambda _: {'same': 'data'}, write_once=r.write_once, engineering_cases=lambda _: engineering,
        execute_case=execute, technical=SimpleNamespace(validate_groups=lambda rows: None),
        run_lanes=lambda rows, operation: ([operation(row, row['index']%4) for row in rows], []),
        study=SimpleNamespace(PLAN_SHA='plan', collect=lambda _, rows: {'status': 'complete', 'count': len(rows)}),
        common=SimpleNamespace(SOURCE='source'))
    if engineering_failure:
        with pytest.raises(RuntimeError, match='engineering failed'): r.supervise_map(module, spec)
        assert not (Path(spec['control'])/'science').exists()
        assert len(seen) == 6
    else:
        assert r.supervise_map(module, spec) == 0
        assert r.read(Path(spec['reports'])/'matched-map-report.json')['count'] == 60
        assert len(seen) == 66
    assert sorted(item[1] for item in seen if item[0]) == list(range(6))
    assert all(item[2] == ('frozen', str(item[1])) for item in seen if item[0])


def test_actual_index_addressed_reference_process_tree_counts_once(tmp_path):
    policy = dict(cohorts={'Q': dict(control='/work/Q'), 'M': dict(control='/work/M')})
    runner = '/workspace/releases/posthoc-reference-e20834c88c10/posthoc_runner.py'
    cases = [(100, 1, ['sh', '-c', 'python '+runner+' run 12']),
        (101, 100, ['python', runner, 'run', '12']),
        (102, 101, ['python', runner.replace('runner', 'entry'), '--seed=10', '--run_group=double_utd20']),
        (103, 1, ['python', runner, 'run', '13'])]
    for pid, parent, argv in cases:
        p = tmp_path/str(pid); p.mkdir(); (p/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv))
        (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0'); (p/'status').write_text(f'PPid:\t{parent}\n')
    assert r.foreign_counts(policy, tmp_path) == [2, 0, 0, 0]


def test_metric_timeout_is_conservative_not_a_training_retry(monkeypatch):
    def timeout(*args, **kwargs): raise r.subprocess.TimeoutExpired('nvidia-smi', 10)
    monkeypatch.setattr(r.subprocess, 'check_output', timeout)
    assert r.resource_probe({}, 0) is False


@pytest.mark.parametrize('complete', [True, False])
def test_verified_engineering_reconciliation_never_calls_driver(tmp_path, monkeypatch, complete):
    from types import SimpleNamespace
    row = dict(index=0, task='triple2', method='full', seed=9826, argv=['frozen','driver'])
    root = tmp_path/'engineering'; r.write_once(root/'cases/case-000.json', row)
    (root/'logs').mkdir()  # Existing completed drivers always have their original log directory.
    paths = {label: root/'processes'/f'case-000-{label}-exit.json' for label in ['driver','validator']}
    for path in paths.values(): r.write_once(path, dict(pid=12, returncode=0))
    record = dict(row, status='complete' if complete else 'running', returncode=0, validation_returncode=0)
    for label,path in paths.items(): record[label+'_exit_path']=str(path); record[label+'_exit_sha256']=r.digest(path)
    r.write_once(root/'runs/case-000.json', record)
    policy = dict(resume_verified_engineering=True, _sha='policy', map_fill_only_unused_slots_until_q_terminal=False)
    module = SimpleNamespace(check=lambda _: None, command=lambda value, mode, *args:[mode,*map(str,args)], gate=lambda _: {},
        environment=lambda value,gpu=None:dict(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES=''),
        execute_case=lambda *args,**kwargs:pytest.fail('completed engineering must never train again'), write_once=r.write_once)
    spec = dict(_root=str(tmp_path), source='source')
    calls=[]
    class Child:
        pid=998
        def __init__(self, argv):
            calls.append(argv); assert argv[0]=='validate'
            r.write_once(Path(argv[argv.index('--output')+1]), {'read_only': True})
        def wait(self): return 0
    monkeypatch.setattr(r.subprocess,'Popen',lambda argv,**kwargs:Child(argv))
    r.install(policy,'M',module,spec)
    if complete:
        assert module.execute_case(spec,row,root,0,engineering=True)==record
        assert len(calls)==1 and r.read(root/'runs/case-000.json')==record
    else:
        with pytest.raises(ValueError,match='cannot be retried'):module.execute_case(spec,row,root,0,engineering=True)
        assert not calls
