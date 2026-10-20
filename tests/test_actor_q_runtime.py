import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from research import actor_q_runtime as r
from research.actor_q_validation import SOURCE, read, write_once, digest


@pytest.fixture
def upstream(tmp_path):
    root=tmp_path/'gac';root.mkdir();spec_path=tmp_path/'upstream.json'
    write_once(spec_path,dict(job_id='job',control=str(root),controller_files={}))
    spec=dict(upstream_control=str(root),upstream_runtime_spec=str(spec_path),
              upstream_runtime_spec_sha256=digest(spec_path),job_id='job')
    return spec,root


def finish(root,code=0):
    write_once(root/'terminal.json',dict(source_commit=r.GAC_SOURCE,expected_runs=32,
        status='complete' if code==0 else 'needs_review',scheduler_exit_observed=True,returncode=code))
    write_once(root/'scheduler-exit.json',dict(returncode=code,started=10.,ended=20.))


def test_waiting_gate_is_read_only(upstream):
    spec,root=upstream;before=list(root.iterdir())
    assert r.gate(spec)==dict(stage='waiting_for_existing_gac_queue')
    assert list(root.iterdir())==before


@pytest.mark.parametrize('code',[0,2,7])
def test_gate_uses_terminal_integrity_without_performance_or_zero_exit_threshold(upstream,code):
    spec,root=upstream;finish(root,code)
    assert r.gate(spec)['upstream_returncode']==code


@pytest.mark.parametrize('problem',['source','count','unknown_exit','bool_exit','unobserved','mismatched_exit','clock','spec'])
def test_invalid_upstream_proof_cannot_release_gate(upstream,problem):
    spec,root=upstream;finish(root)
    value=read(root/'terminal.json');observed=read(root/'scheduler-exit.json')
    if problem=='source':value['source_commit']='other'
    elif problem=='count':value['expected_runs']=31
    elif problem=='unknown_exit':value['returncode']=None
    elif problem=='bool_exit':value['returncode']=False
    elif problem=='unobserved':value['scheduler_exit_observed']=False
    elif problem=='mismatched_exit':observed['returncode']=2
    elif problem=='clock':observed['ended']=1.
    else:Path(spec['upstream_runtime_spec']).write_text('{}')
    (root/'terminal.json').write_text(json.dumps(value));(root/'scheduler-exit.json').write_text(json.dumps(observed))
    with pytest.raises((ValueError,KeyError)):r.gate(spec)


def test_gpu_probe_requires_four_4090s_and_no_live_compute_pids(monkeypatch):
    names='NVIDIA RTX 4090\n'*4
    monkeypatch.setattr(r.subprocess,'check_output',lambda argv,**_:names if '--query-gpu=name' in argv else '')
    assert r.idle_gpus()
    monkeypatch.setattr(r.subprocess,'check_output',lambda argv,**_:names if '--query-gpu=name' in argv else '123\n')
    assert not r.idle_gpus()
    monkeypatch.setattr(r.subprocess,'check_output',lambda *_,**__:'NVIDIA RTX6000\n'*4)
    with pytest.raises(ValueError):r.idle_gpus()


def test_arm_persists_intent_and_refuses_second_submission(tmp_path,monkeypatch):
    spec=dict(control=str(tmp_path/'control'),_root=str(tmp_path),_spec_path='spec',_spec_sha256='sha',python='python')
    monkeypatch.setattr(r,'check',lambda _:None);calls=[]
    def popen(argv,**kwargs):calls.append(argv);return SimpleNamespace(pid=4321)
    monkeypatch.setattr(r.subprocess,'Popen',popen)
    assert r.arm(spec)['pid']==4321
    with pytest.raises(FileExistsError):r.arm(spec)
    assert len(calls)==1 and read(tmp_path/'control/continuation/arm-intent.json')['automatic_resubmission'] is False


def test_unknown_submission_preserves_intent_and_never_resubmits(tmp_path,monkeypatch):
    spec=dict(control=str(tmp_path/'control'),_root=str(tmp_path),_spec_path='spec',_spec_sha256='sha',python='python')
    monkeypatch.setattr(r,'check',lambda _:None);calls=[]
    def unknown(*_,**__):calls.append(1);raise OSError('Unknown launch response')
    monkeypatch.setattr(r.subprocess,'Popen',unknown)
    with pytest.raises(OSError):r.arm(spec)
    with pytest.raises(FileExistsError):r.arm(spec)
    assert len(calls)==1


def test_waiting_owner_does_not_probe_or_allocate_gpus_or_create_science(tmp_path,monkeypatch):
    spec=dict(control=str(tmp_path),_spec_sha256='sha')
    monkeypatch.setattr(r,'check',lambda _:None)
    monkeypatch.setattr(r,'gate',lambda _:dict(stage='waiting_for_existing_gac_queue'))
    def forbidden():raise AssertionError('GPU probe during upstream wait')
    monkeypatch.setattr(r,'idle_gpus',forbidden)
    def stop(_):raise StopIteration()
    with pytest.raises(StopIteration):r.own(spec,sleep=stop)
    assert not (tmp_path/'science').exists() and not (tmp_path/'engineering-gpu-r1').exists()
    assert read(tmp_path/'continuation/status.json')['stage']=='waiting_for_existing_gac_queue'


def test_four_sequential_lanes_visit_every_identity_once_and_keep_failed_runs():
    rows=[dict(index=i) for i in range(40)];active=[0]*4;maximum=[0]*4;seen=[];lock=threading.Lock()
    def operation(row,gpu):
        with lock:active[gpu]+=1;maximum[gpu]=max(maximum[gpu],active[gpu]);seen.append(row['index'])
        time.sleep(.001)
        with lock:active[gpu]-=1
        return dict(index=row['index'],status='failed' if row['index']==8 else 'complete')
    output,errors=r.run_lanes(rows,operation)
    assert sorted(seen)==list(range(40)) and maximum==[1]*4 and not errors
    assert next(v for v in output if v['index']==8)['status']=='failed'


def test_infrastructure_exception_never_restarts_case_or_fills_missing_lane():
    seen=[]
    def operation(row,gpu):
        seen.append(row['index'])
        if row['index']==8:raise OSError('Launch unknown')
        return row
    result,errors=r.run_lanes([dict(index=i) for i in range(40)],operation)
    assert len(seen)==len(set(seen)) and seen.count(8)==1
    assert len(errors)==1 and errors[0]['incident_index']==8 and 12 not in seen


def test_engineering_changes_only_declared_operational_fields():
    cases=[]
    for task in ('triple2','double2'):
        for method in ('full','remove_direction_radial'):
            cases.append(dict(index=len(cases),task=task,method=method,seed=20270,run_group=task+method,
                argv=['python','main_online.py','--seed=20270','--online_steps=1000000','--start_training=5000',
                      '--buffer_size=1000000','--eval_interval=100000','--eval_episodes=50','--save_interval=1000000',
                      '--log_interval=5000','--run_group='+task+method,'--save_dir=/workspace/original',
                      '--agent.adapter_actor_q_gradient='+method,'--agent.target_entropy=-3.0']))
    plan=dict(tasks=['triple2','double2'],methods=['full','remove_direction_radial'],cases=cases)
    result=r.engineering_cases(plan)
    assert len(result)==4 and all(v['seed']==9823 for v in result)
    for original,short in zip(cases,result,strict=True):
        a,b=map(r.validation.flags_from_argv,(original['argv'],short['argv']))
        assert a['agent.adapter_actor_q_gradient']==b['agent.adapter_actor_q_gradient']
        assert a['agent.target_entropy']==b['agent.target_entropy']=='-3.0'
        assert b['online_steps']=='64' and b['start_training']=='32'


def test_actual_supervisor_exit_is_observed_before_terminal(tmp_path,monkeypatch):
    spec=dict(control=str(tmp_path/'control'),reports=str(tmp_path/'reports'),_root=str(tmp_path),
        _spec_path='spec',_spec_sha256='sha',python='python')
    monkeypatch.setattr(r,'check',lambda _:None);monkeypatch.setattr(r,'gate',lambda _:dict(stage='upstream_terminal'))
    monkeypatch.setattr(r,'idle_gpus',lambda:True)
    class Child:
        pid=1234
        def wait(self):
            assert not (tmp_path/'control/continuation/terminal.json').exists()
            return 2
    monkeypatch.setattr(r.subprocess,'Popen',lambda *_,**__:Child())
    assert r.own(spec)==2
    terminal=read(tmp_path/'control/continuation/terminal.json')
    assert terminal['status']=='needs_review' and terminal['scheduler_exit_observed'] and terminal['returncode']==2
    assert read(tmp_path/'control/continuation/supervisor-exit.json')['returncode']==2


@pytest.mark.parametrize('driver_code,validation_code',[(0,0),(1,None),(0,1)])
def test_case_preserves_actual_exits_and_never_retries_existing_attempt(tmp_path,monkeypatch,driver_code,validation_code):
    spec=dict(_root=str(tmp_path),python='python',source='source',source_binding='binding',_spec_path='spec')
    row=dict(index=0,task='triple2',method='full',seed=20270,argv=['python','main_online.py'])
    monkeypatch.setattr(r,'check',lambda _:None);calls=[]
    class Child:
        def __init__(self,argv):
            self.argv=argv;self.pid=1000+len(calls);calls.append(argv)
        def wait(self):
            if '--output' in self.argv:
                output=Path(self.argv[self.argv.index('--output')+1])
                if validation_code==0:
                    write_once(output,dict(index=0,task='triple2',method='full',seed=20270,source_commit=SOURCE,status='complete'))
                return validation_code
            initial=Path(self.argv[self.argv.index('--initial-receipt')+1]);write_once(initial,{'observed':True})
            return driver_code
    monkeypatch.setattr(r.subprocess,'Popen',lambda argv,**_:Child(argv))
    output=r.execute_case(spec,row,tmp_path/'attempt',0)
    assert output['returncode']==driver_code and output['driver_exit_observed'] is True
    assert output['status']==('complete' if driver_code==validation_code==0 else 'failed' if driver_code else 'needs_review')
    assert len(calls)==(1 if driver_code else 2)
    with pytest.raises(FileExistsError):r.execute_case(spec,row,tmp_path/'attempt',0)
    assert len(calls)==(1 if driver_code else 2)


def test_unknown_driver_wait_is_retained_without_a_second_launch(tmp_path,monkeypatch):
    spec=dict(_root=str(tmp_path),python='python',source='source',source_binding='binding',_spec_path='spec')
    row=dict(index=0,task='triple2',method='full',seed=20270,argv=['python','main_online.py']);calls=[]
    monkeypatch.setattr(r,'check',lambda _:None)
    class Child:
        pid=4567
        def wait(self):raise OSError('Exit observation unavailable')
    def launch(*_,**__):calls.append(1);return Child()
    monkeypatch.setattr(r.subprocess,'Popen',launch)
    with pytest.raises(OSError):r.execute_case(spec,row,tmp_path/'attempt',0)
    record=read(tmp_path/'attempt/runs/case-000.json')
    assert record['status']=='running' and 'returncode' not in record
    with pytest.raises(FileExistsError):r.execute_case(spec,row,tmp_path/'attempt',0)
    assert calls==[1]
