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


def test_cross_process_slot_limit_reserves_existing_work_and_staggers(tmp_path):
    policy = dict(admission_root=str(tmp_path), gpu_tasks=3, reserved_external_slots=1, stagger_seconds=.025)
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
