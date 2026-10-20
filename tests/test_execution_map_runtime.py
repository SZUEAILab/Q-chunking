import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from research import execution_map_runtime as r
from research.actor_q_validation import SOURCE, read, write_once, digest


@pytest.fixture
def upstream(tmp_path,monkeypatch):
    root=tmp_path/'actor-q';root.mkdir();continuation=root/'continuation';continuation.mkdir();spec_path=tmp_path/'upstream.json'
    original=tmp_path/'plan.json';original.write_text('frozen original forty')
    write_once(spec_path,dict(job_id='job',control=str(root),implementation_files={},plan=str(original),plan_sha256=r.common.PLAN_SHA))
    spec=dict(upstream_control=str(root),upstream_runtime_spec=str(spec_path),
              upstream_runtime_spec_sha256=digest(spec_path),job_id='job')
    original_digest=r.digest
    monkeypatch.setattr(r,'digest',lambda p:r.common.PLAN_SHA if Path(p)==original else original_digest(p))
    return spec,continuation


def finish(root,code=0):
    write_once(root/'terminal.json',dict(source_commit=SOURCE,expected_runs=40,spec_sha256='upstream-sha',
        status='complete' if code==0 else 'needs_review',scheduler_exit_observed=True,returncode=code))
    write_once(root/'supervisor-exit.json',dict(returncode=code,started=10.,ended=20.))


def test_waiting_gate_is_read_only(upstream):
    spec,root=upstream;before=list(root.iterdir())
    assert r.gate(spec)==dict(stage='waiting_for_existing_actor_q_queue')
    assert list(root.iterdir())==before


@pytest.mark.parametrize('code',[0,2,7])
def test_gate_uses_terminal_integrity_without_performance_or_zero_exit_threshold(upstream,code):
    spec,root=upstream;finish(root,code);value=read(root/'terminal.json');value['spec_sha256']=spec['upstream_runtime_spec_sha256'];(root/'terminal.json').write_text(json.dumps(value))
    assert r.gate(spec)['upstream_returncode']==code


@pytest.mark.parametrize('problem',['source','count','unknown_exit','bool_exit','unobserved','mismatched_exit','clock','spec'])
def test_invalid_upstream_proof_cannot_release_gate(upstream,problem):
    spec,root=upstream;finish(root);value=read(root/'terminal.json');value['spec_sha256']=spec['upstream_runtime_spec_sha256'];(root/'terminal.json').write_text(json.dumps(value))
    value=read(root/'terminal.json');observed=read(root/'supervisor-exit.json')
    if problem=='source':value['source_commit']='other'
    elif problem=='count':value['expected_runs']=39
    elif problem=='unknown_exit':value['returncode']=None
    elif problem=='bool_exit':value['returncode']=False
    elif problem=='unobserved':value['scheduler_exit_observed']=False
    elif problem=='mismatched_exit':observed['returncode']=2
    elif problem=='clock':observed['ended']=1.
    else:Path(spec['upstream_runtime_spec']).write_text('{}')
    (root/'terminal.json').write_text(json.dumps(value));(root/'supervisor-exit.json').write_text(json.dumps(observed))
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
    monkeypatch.setattr(r,'gate',lambda _:dict(stage='waiting_for_existing_actor_q_queue'))
    def forbidden():raise AssertionError('GPU probe during upstream wait')
    monkeypatch.setattr(r,'idle_gpus',forbidden)
    def stop(_):raise StopIteration()
    with pytest.raises(StopIteration):r.own(spec,sleep=stop)
    assert not (tmp_path/'science').exists() and not (tmp_path/'engineering-gpu-r1').exists()
    assert read(tmp_path/'continuation/status.json')['stage']=='waiting_for_existing_actor_q_queue'


def test_four_sequential_lanes_visit_every_identity_once_and_keep_failed_runs():
    rows=[dict(index=i) for i in range(60)];active=[0]*4;maximum=[0]*4;seen=[];lock=threading.Lock()
    def operation(row,gpu):
        with lock:active[gpu]+=1;maximum[gpu]=max(maximum[gpu],active[gpu]);seen.append(row['index'])
        time.sleep(.001)
        with lock:active[gpu]-=1
        return dict(index=row['index'],status='failed' if row['index']==8 else 'complete')
    output,errors=r.run_lanes(rows,operation)
    assert sorted(seen)==list(range(60)) and maximum==[1]*4 and not errors
    assert next(v for v in output if v['index']==8)['status']=='failed'


def test_infrastructure_exception_never_restarts_case_or_fills_missing_lane():
    seen=[]
    def operation(row,gpu):
        seen.append(row['index'])
        if row['index']==8:raise OSError('Launch unknown')
        return row
    result,errors=r.run_lanes([dict(index=i) for i in range(60)],operation)
    assert len(seen)==len(set(seen)) and seen.count(8)==1
    assert len(errors)==1 and errors[0]['incident_index']==8 and 12 not in seen


def test_engineering_changes_only_declared_operational_fields():
    cases=[]
    for task in ('triple2','double2'):
        for mapping in ('direction_speed','cartesian_aux','cartesian_projection_aux'):
            method='full'
            cases.append(dict(index=len(cases),task=task,method=method,execution_map=mapping,seed=20290,run_group=task+mapping,
                argv=['python','main_online.py','--seed=20290','--online_steps=1000000','--start_training=5000',
                      '--buffer_size=1000000','--eval_interval=100000','--eval_episodes=50','--save_interval=1000000',
                      '--log_interval=5000','--run_group='+task+mapping,'--save_dir=/workspace/original',
                      '--agent.adapter_actor_q_gradient='+method,'--agent.target_entropy=-3.0',
                      '--ds_adapter_execution_map='+mapping,'--ds_adapter_critic_space=env']))
    plan=dict(tasks=['triple2','double2'],maps=['direction_speed','cartesian_aux','cartesian_projection_aux'],cases=cases)
    result=r.engineering_cases(plan)
    assert len(result)==6 and all(v['seed']==9826 for v in result)
    for original,short in zip(cases,result,strict=True):
        a,b=map(r.common.flags_from_argv,(original['argv'],short['argv']))
        assert a['agent.adapter_actor_q_gradient']==b['agent.adapter_actor_q_gradient']
        assert a['agent.target_entropy']==b['agent.target_entropy']=='-3.0'
        assert b['online_steps']=='64' and b['start_training']=='32'
        assert a['ds_adapter_execution_map']==b['ds_adapter_execution_map']==short['execution_map']
        assert a['ds_adapter_critic_space']==b['ds_adapter_critic_space']=='env'


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
    spec=dict(_root=str(tmp_path),python='python',source='source',source_binding='binding',expected_gpu='4090',_spec_path='spec')
    row=dict(index=0,task='triple2',method='full',execution_map='direction_speed',seed=20270,argv=['python','main_online.py'])
    monkeypatch.setattr(r,'check',lambda _:None);calls=[]
    class Child:
        def __init__(self,argv):
            self.argv=argv;self.pid=1000+len(calls);calls.append(argv)
        def wait(self):
            if '--output' in self.argv:
                output=Path(self.argv[self.argv.index('--output')+1])
                if validation_code==0:
                    write_once(output,dict(index=0,task='triple2',method='full',execution_map='direction_speed',seed=20270,source_commit=SOURCE,status='complete'))
                return validation_code
            initial=Path(self.argv[self.argv.index('--initial-receipt')+1]);write_once(initial,{'observed':True})
            return driver_code
    monkeypatch.setattr(r.subprocess,'Popen',lambda argv,**_:Child(argv))
    output=r.execute_case(spec,row,tmp_path/'attempt',0)
    assert output['returncode']==driver_code and output['driver_exit_observed'] is True
    assert output['status']==('complete' if driver_code==validation_code==0 else 'failed' if driver_code else 'needs_review')
    assert len(calls)==(1 if driver_code else 2)
    r.study.verify_terminal_receipts([output])
    with pytest.raises(FileExistsError):r.execute_case(spec,row,tmp_path/'attempt',0)
    assert len(calls)==(1 if driver_code else 2)


def test_unknown_driver_wait_is_retained_without_a_second_launch(tmp_path,monkeypatch):
    spec=dict(_root=str(tmp_path),python='python',source='source',source_binding='binding',expected_gpu='4090',_spec_path='spec')
    row=dict(index=0,task='triple2',method='full',execution_map='direction_speed',seed=20270,argv=['python','main_online.py']);calls=[]
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


def test_missing_upstream_owner_without_terminal_never_releases_queue(upstream,monkeypatch):
    spec,root=upstream
    write_once(root/'started.json',dict(pid=123,spec_sha256=spec['upstream_runtime_spec_sha256']))
    monkeypatch.setattr(r,'alive',lambda *_:False)
    with pytest.raises(RuntimeError,match='owner missing'):r.gate(spec)


def test_upstream_owner_is_verified_from_actual_process_arguments(upstream,monkeypatch):
    spec,root=upstream;seen=[]
    write_once(root/'started.json',dict(pid=123,spec_sha256=spec['upstream_runtime_spec_sha256']))
    def alive(*args):seen.append(args);return True
    monkeypatch.setattr(r,'alive',alive)
    assert r.gate(spec)['stage']=='waiting_for_existing_actor_q_queue'
    assert seen==[(123,spec['upstream_runtime_spec'],'own')]


def test_engineering_failure_does_not_create_any_scientific_run(tmp_path,monkeypatch):
    spec=dict(control=str(tmp_path),_spec_sha256='spec')
    monkeypatch.setattr(r,'check',lambda _:dict())
    monkeypatch.setattr(r,'gate',lambda _:dict(stage='upstream_terminal'))
    monkeypatch.setattr(r,'idle_gpus',lambda:True)
    monkeypatch.setattr(r,'verify_data',lambda _:dict())
    monkeypatch.setattr(r,'engineering_cases',lambda _:[dict(index=0)])
    monkeypatch.setattr(r,'execute_case',lambda *_,**__:dict(status='failed'))
    with pytest.raises(RuntimeError,match='engineering failed'):r.supervise(spec)
    assert not (tmp_path/'science').exists()


def test_independent_validator_adds_all_bound_artifacts_and_original_case_identity(tmp_path,monkeypatch):
    directory=tmp_path/'run';directory.mkdir()
    for name in ('flags.json','env.csv','online_agent.csv','eval.csv','params_64.pkl'):
        (directory/name).write_text('synthetic bounded artifact')
    write_once(directory/'effective_agent_config.json',dict(adapter_execution_map='cartesian_projection_aux'))
    case=tmp_path/'case.json';initial=tmp_path/'initial.json';output=tmp_path/'validation.json'
    row=dict(index=4,task='double2',seed=9826,method='full',execution_map='cartesian_projection_aux',
        argv=['python','main_online.py','--online_steps=64'])
    write_once(case,row);write_once(initial,dict(source_binding_sha256='bound'))
    spec=dict(source=str(tmp_path/'source'),source_binding='binding',source_binding_sha256='bound',expected_gpu='4090')
    monkeypatch.setattr(r.technical,'validate_run',lambda *_,**__:dict(run_dir=str(directory),status='complete'))
    monkeypatch.setattr(r.sys,'path',list(r.sys.path))
    value=r.validate_case(spec,case,initial,output)
    assert all(value[k]==row[k] for k in ('index','task','seed','method','execution_map'))
    assert set(value['identity_hashes'])=={'flags.json','effective_agent_config.json','env.csv','online_agent.csv','eval.csv','params_64.pkl'}
    assert value['source_binding_sha256']=='bound'
    assert value['effective_config']==read(directory/'effective_agent_config.json')


@pytest.mark.parametrize('field,value',[('format','other'),('job_id','new-job'),
    ('expected_gpu','RTX6000 Ada'),('python','python'),('plan_sha256','other'),
    ('source_binding_sha256','other'),('upstream_runtime_spec_sha256','other'),
    ('package_versions',{}),('implementation_files',{}),('control','/workspace/control/other'),
    ('reports','/workspace/reports/other')])
def test_load_spec_rejects_different_resource_plan_or_runtime_before_side_effects(tmp_path,monkeypatch,field,value):
    spec=dict(format='matched-map-runtime-v1',job_id='a21566001929515008267857',
        implementation_files={name:'sha' for name in r.FILES},plan_sha256=r.study.PLAN_SHA,
        expected_gpu='4090',python='/workspace/venv/bin/python',source_binding_sha256=r.study.SOURCE_BINDING_SHA,
        upstream_runtime_spec_sha256=r.UPSTREAM_SPEC_SHA,package_versions=r.VERSIONS,
        control='/workspace/control/'+r.study.COHORT,reports='/workspace/reports/'+r.study.COHORT)
    spec[field]=value;path=tmp_path/'spec.json';write_once(path,spec)
    def forbidden(_):raise AssertionError('Invalid specification reached runtime')
    monkeypatch.setattr(r,'check',forbidden)
    with pytest.raises(ValueError):r.load_spec(path)
