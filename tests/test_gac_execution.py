import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from research import gac_execution as execution


FAKE_DRIVER = '''
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output-dir');args=p.parse_args()
plan=json.loads(Path(args.plan).read_text());directory=Path(args.output_dir);directory.mkdir()
(directory/'fixture-output.json').write_text(json.dumps({'fixture_only':True,'seed':plan['seed']}))
raise SystemExit(plan.get('fixture_exit',0))
'''
FAKE_VALIDATOR = '''
from pathlib import Path
import hashlib,json
def validate_run(directory,plan_path,process,source):
 plan=json.loads(Path(plan_path).read_text());raw=json.loads((directory/'fixture-output.json').read_text())
 assert raw['fixture_only'] and raw['seed']==plan['seed'] and process['returncode']==0
 if plan.get('fixture_validation_failure'):raise ValueError('deliberate fixture validation failure')
 return dict(status='complete',returncode=0,source_commit=plan['source_commit'],seed=plan['seed'],
  env_name=plan['env_name'],plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
  online_steps=plan['online_steps'],runtime={'device':plan.get('fixture_backend','cuda')})
'''


def fixture_case(tmp_path, **updates):
    source=tmp_path/'source';source.mkdir()
    driver=source/'main_gac_reference.py';driver.write_text(FAKE_DRIVER)
    validator=tmp_path/'fixture_validator.py';validator.write_text(FAKE_VALIDATOR)
    science=tmp_path/'science';science.mkdir()
    plan=dict(source_commit=execution.SOURCE,source_files={'main_gac_reference.py':execution.digest(driver)},
              seed=9601,env_name='fixture-no-simulator',replay_protocol='online_only',device='cuda',online_steps=64)
    plan.update(updates);path=science/'plan.json';path.write_text(json.dumps(plan))
    row=dict(name='engineering-fixture',plan='plan.json',plan_sha256=execution.digest(path),seed=9601,
             env_name=plan['env_name'],protocol='online_only',profile='r1',task='fixture',gpu=2,position=0)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONDONTWRITEBYTECODE='1')
    return row,dict(science=science,source=source,validator=validator,validator_sha256=execution.digest(validator),
                    python=sys.executable,control=tmp_path/'control',results=tmp_path/'results',
                    environment=env,expected_source_files=1)


def test_real_subprocesses_record_distinct_driver_and_validator_exits(tmp_path):
    row,kwargs=fixture_case(tmp_path)
    result=execution.execute_case(row,**kwargs)
    assert result['status']=='complete' and result['returncode']==0 and result['validation_returncode']==0
    assert result['pid']!=result['validation_pid'] and result['ended']<=result['validation_started']<=result['validation_ended']
    assert result['gpu']==2 and result['online_steps']==64
    assert result['validation_source_sha256']==kwargs['validator_sha256']
    assert execution.read(kwargs['control']/'runs/engineering-fixture.json')==result
    process=execution.read(kwargs['control']/'processes/engineering-fixture.json')
    assert process['returncode']==0 and process['pid']==result['pid']
    # The fixture runs plain Python only, with no simulator, framework or GPU work.
    assert execution.read(kwargs['results']/'engineering-fixture/fixture-output.json')['fixture_only']


def test_driver_failure_is_retained_and_never_validated_or_restarted(tmp_path):
    row,kwargs=fixture_case(tmp_path,fixture_exit=7)
    result=execution.execute_case(row,**kwargs)
    assert result['status']=='failed' and result['returncode']==7
    assert 'validation_returncode' not in result
    assert not (kwargs['control']/'validations/engineering-fixture.json').exists()
    with pytest.raises(FileExistsError):execution.execute_case(row,**kwargs)
    assert execution.read(kwargs['control']/'runs/engineering-fixture.json')['pid']==result['pid']


def test_independent_validation_failure_keeps_its_observed_exit(tmp_path):
    row,kwargs=fixture_case(tmp_path,fixture_validation_failure=True)
    result=execution.execute_case(row,**kwargs)
    assert result['status']=='needs_review' and result['returncode']==0 and result['validation_returncode']!=0
    assert 'deliberate fixture validation failure' in (kwargs['control']/'logs/engineering-fixture-validation.log').read_text()
    with pytest.raises(FileExistsError):execution.execute_case(row,**kwargs)


def test_validation_backend_mismatch_preserves_both_known_exits(tmp_path):
    row,kwargs=fixture_case(tmp_path,fixture_backend='cpu')
    with pytest.raises(ValueError,match='backend'):execution.execute_case(row,**kwargs)
    result=execution.read(kwargs['control']/'runs/engineering-fixture.json')
    assert result['status']=='needs_review' and result['returncode']==0 and result['validation_returncode']==0
    assert result['validation_pid']!=result['pid'] and 'lifecycle_error' in result


@pytest.mark.parametrize('change',['plan','source','validator','seed','gpu','existing_directory','existing_process'])
def test_mismatched_sources_and_previous_attempts_cannot_start_a_process(tmp_path,monkeypatch,change):
    row,kwargs=fixture_case(tmp_path)
    if change=='plan':(kwargs['science']/'plan.json').write_text('{}')
    elif change=='source':(kwargs['source']/'main_gac_reference.py').write_text('changed')
    elif change=='validator':kwargs['validator'].write_text('changed')
    elif change=='seed':row['seed']=9602
    elif change=='gpu':kwargs['environment']['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    elif change=='existing_directory':(kwargs['results']/row['name']).mkdir(parents=True)
    else:
        path=kwargs['control']/'processes'/(row['name']+'.json');path.parent.mkdir(parents=True);path.write_text('prior attempt')
    monkeypatch.setattr(execution.subprocess,'Popen',lambda *a,**k:pytest.fail('A forbidden process was started'))
    with pytest.raises((ValueError,FileExistsError)):execution.execute_case(row,**kwargs)


def test_lost_driver_wait_preserves_unknown_exit_instead_of_claiming_failure(tmp_path,monkeypatch):
    row,kwargs=fixture_case(tmp_path)
    class LostWait:
        pid=999999999
        def wait(self):raise OSError('fixture lost observation')
    monkeypatch.setattr(execution.subprocess,'Popen',lambda *a,**k:LostWait())
    with pytest.raises(OSError):execution.execute_case(row,**kwargs)
    record=execution.read(kwargs['control']/'runs/engineering-fixture.json')
    assert record['status']=='needs_review' and 'returncode' not in record and record['pid']==999999999
    with pytest.raises(FileExistsError):execution.execute_case(row,**kwargs)


def lane_rows():
    return [dict(name=f'fixture-gpu{gpu}-position{position}',gpu=gpu,position=position,seed=10+gpu)
            for gpu in range(4) for position in range(8)]


def test_four_lanes_never_overlap_work_on_the_same_gpu_and_keep_queue_order():
    lock=threading.Lock();barrier=threading.Barrier(4);active=set();observed={gpu:[] for gpu in range(4)};peak=[]
    def case(row):
        gpu=row['gpu']
        with lock:
            assert gpu not in active;active.add(gpu);peak.append(len(active));observed[gpu].append(row['position'])
        if row['position']==0:barrier.wait(timeout=5)
        time.sleep(.001)
        with lock:active.remove(gpu)
        return dict(row,status='complete')
    result=execution.run_lanes(list(reversed(lane_rows())),case)
    assert not result['lane_errors'] and len(result['records'])==32 and max(peak)==4
    assert all(positions==list(range(8)) for positions in observed.values())


def test_driver_failure_does_not_selectively_cancel_other_registered_conditions():
    def case(row):return dict(row,status='failed' if row['position']==0 else 'complete')
    result=execution.run_lanes(lane_rows(),case)
    assert len(result['records'])==32 and not result['lane_errors']
    assert sum(r['status']=='failed' for r in result['records'])==4


def test_infrastructure_exception_stops_only_its_lane_and_keeps_unstarted_identities():
    def case(row):
        if row['gpu']==2 and row['position']==3:raise OSError('fixture source/transport incident')
        return dict(row,status='complete')
    result=execution.run_lanes(lane_rows(),case)
    assert len(result['records'])==27 and len(result['lane_errors'])==1
    error=result['lane_errors'][0]
    assert error['gpu']==2 and error['incident_run']=='fixture-gpu2-position3'
    assert error['not_started']==[f'fixture-gpu2-position{i}' for i in range(4,8)]


@pytest.mark.parametrize('change',['duplicate','missing','order','seed'])
def test_invalid_lane_matrix_cannot_invoke_execution(change):
    rows=lane_rows()
    if change=='duplicate':rows[1]=dict(rows[0])
    elif change=='missing':rows.pop()
    elif change=='order':rows[1]['position']=0
    else:rows[1]['seed']=14
    with pytest.raises(ValueError):execution.run_lanes(rows,lambda _:pytest.fail('Invalid matrix started'))
