import copy
import importlib.util
import json
from pathlib import Path

import pytest

from research import gac_runtime as runtime


def helper(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location(name+'_fixtures',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


design=helper('test_gac_comparisons')
fql=helper('test_gac_environment_runtime')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';science=bundle/'science';science.mkdir(parents=True)
    cohort,rows=design.save_science_fixture(science)
    cohort.update(replay_protocols=['online_only','prior_balanced'], profiles=[{'name':'r1'},{'name':'r25'}], job_id='fixture-job')
    save(science/'cohort.json',cohort)
    binding=runtime.read(science/'freeze-binding.json');binding['cohort_sha256']=runtime.digest(science/'cohort.json')
    save(science/'freeze-binding.json',binding)
    # This synthetic fixture exercises receipt logic only, never a real learner.
    env=dict(control=str(tmp_path/'environment-control'),prefix=str(tmp_path/'prefix'),job_id='fixture-job',
             runtime_root=str(bundle),dependency_receipt=str(tmp_path/'dependency-receipt.json'),
             environment_tool_sha256=runtime.digest(runtime.environment.__file__))
    for name,value in zip(('gate_terminal','gate_report','gate_scheduler_exit'),fql.completed_fql()):
        path=tmp_path/'fql'/name;save(path,value);env[name]=str(path)
    env_path=tmp_path/'environment-spec.json';save(env_path,env)
    save(env['dependency_receipt'],{'fixture_only':True})
    control=Path(env['control'])
    spec=dict(format='gac-runtime-spec-v1',environment_spec=str(env_path),job_id='fixture-job',team_id='fixture-team',
              cluster_id='fixture-cluster',environment_ready=str(control/'ready-for-image.json'),
              image_receipt=str(control/'image-ready.json'),control=str(tmp_path/'science-control'),
              results=str(tmp_path/'science-results'),reports=str(tmp_path/'science-reports'),
              _spec_sha256='fixture-spec-sha',_spec_path=str(tmp_path/'spec.json'),_tools_root=str(tmp_path))
    manifest=dict(training_python_sources={'agents/gac_reference.py':'a'*64,'research/check_gac_reference.py':'b'*64},
                  files={'official/GAC.py':'c'*64},upstream_commit='pinned-official-fixture')
    records=[]
    for index,case in enumerate(runtime.environment.engineering_plans(bundle)):
        name=case['name'];plan_path=control/'plans'/(name+'.json');save(plan_path,case['plan'])
        driver=dict(pid=1000+index,returncode=0,started=1.,ended=2.,argv=[str(Path(env['prefix'])/'bin/python')],
                    plan_sha256=runtime.digest(plan_path))
        validation=dict(pid=2000+index,returncode=0,started=3.,ended=4.,argv=list(driver['argv']))
        save(control/'processes'/(name+'-bound.json'),driver)
        save(control/'processes'/(name+'-validation.json'),validation)
        path=control/'validations'/(name+'.json')
        save(path,dict(status='complete',returncode=0,source_commit=runtime.execution.SOURCE,seed=case['plan']['seed'],
             env_name=case['plan']['env_name'],online_steps=64,plan_sha256=driver['plan_sha256'],
             actual_updates=dict(critic=31,actor=30,target=31),exact_checkpoint_fields=179,runtime={'device':'cuda'}))
        records.append(dict(name=name,gpu=case['gpu'],env_name=case['plan']['env_name'],driver=driver,validation=validation,
             initial_model_and_rng_sha256='same-per-task-fixture',report=str(path),report_sha256=runtime.digest(path)))
    ready=dict(status='cloud_engineering_ready_for_image',prefix=env['prefix'],source_commit=runtime.execution.SOURCE,
               analysis_commit=runtime.ANALYSIS,spec_sha256=runtime.digest(env_path),upstream_gate=runtime.environment.inspect_gate(env),
               expected_runs=8,validated_runs=8,engineering_steps_per_run=64,numerical_cases=4,engineering_runs=records,finished=100.)
    save(spec['environment_ready'],ready)
    save(control/'environment.json',dict(status='passed',prefix=env['prefix'],torch='2.14.0+cu126',compiled_cuda='12.6',
         gpus=['NVIDIA RTX 4090']*4,inherited_system_site_packages=False,jax_installed=False,flax_installed=False,
         cuda_libraries=[str(Path(env['prefix'])/'lib/libcublas.so.12')]))
    save(control/'numerical-default.json',dict(status='passed',backend='cuda',local_source_sha256='a'*64,oracle_sha256='b'*64,
         upstream_commit=manifest['upstream_commit'],upstream_source_sha256='c'*64,
         cases=[dict(status='passed',device='cuda',max_absolute_state_error=0)]*3))
    save(control/'numerical-radius25.json',dict(status='passed',device='cuda',max_absolute_state_error=0,config={'radius':2.5},
         local_learner_sha256='a'*64,checker_sha256='b'*64,upstream_source_sha256='c'*64))
    for name in ('create-venv','offline-install','pip-check','four-gpu-environment','source-tests','analysis-tests','numerical-default','numerical-radius25'):
        save(control/'processes'/(name+'.json'),dict(pid=3000,returncode=0))
    request=dict(name='gac-fixture',tag='fixture-v1',jobId=spec['job_id'],teamId=spec['team_id'],clusterId=spec['cluster_id'],
                 uploadType=4,sceneList=['training','training-debug'])
    detail=dict(id='fixture-image',name=request['name'],tag=request['tag'],jobId=spec['job_id'],teamId=spec['team_id'],
                clusterId=spec['cluster_id'],status=2,path='registry.invalid/team/gac-fixture:fixture-v1')
    usable={key:value for key,value in detail.items() if key!='jobId'}
    image=dict(format='gac-saved-image-v1',status='usable',source_job_id=spec['job_id'],prefix=env['prefix'],environment_ready_sha256=runtime.digest(spec['environment_ready']),
               save_requested_at=101.,save_request=request,detail=detail,usable_record=usable,
               detail_api_path='/gateway/foundation/api/v1/image/fixture-image',usable_api_path='/gateway/foundation/api/v1/image/action/page')
    save(spec['image_receipt'],image)
    reports=design.fixture_reports(rows)
    for key,name in [('native_report',runtime.comparisons.NATIVE),('posthoc_report',runtime.comparisons.POSTHOC)]:
        spec[key]=str(tmp_path/(key+'.json'));save(spec[key],reports[name])
    (bundle/'manifest.json').write_text('fixture runtime manifest')
    monkeypatch.setattr(runtime,'verified_runtime',lambda *_:(bundle,manifest,cohort,rows))
    return spec,env,ready,image,rows


def test_ready_gate_binds_actual_paths_and_has_no_performance_threshold(prepared):
    spec,env,*_=prepared
    result=runtime.inspect(spec)
    assert result['status']=='ready_to_launch' and result['expected_runs']==32 and result['performance_threshold'] is None
    assert env['gate_terminal'] in result['input_hashes'] and 'gate_terminal' not in result['input_hashes']
    assert result['image']['new_instance_pull_verified'] is False
    assert all(Path(path).is_file() for path in result['input_hashes'])


@pytest.mark.parametrize('missing,status',[('fql','waiting_for_fql'),('dependency','waiting_for_dependency_transfer'),
    ('engineering','waiting_for_cloud_engineering'),('image','waiting_for_image')])
def test_missing_prerequisite_is_read_only_and_launch_never_creates_scientific_paths(prepared,missing,status):
    spec,env,*_=prepared
    path={'fql':env['gate_terminal'],'dependency':env['dependency_receipt'],
          'engineering':spec['environment_ready'],'image':spec['image_receipt']}[missing]
    Path(path).unlink()
    assert runtime.inspect(spec)['status']==status
    with pytest.raises(RuntimeError,match='deferred'):runtime.launch(spec)
    assert not any(Path(spec[key]).exists() for key in ('control','results','reports'))


@pytest.mark.parametrize('change',['branch_missing','duplicate_branch','driver_unknown','validator_boolean_exit','same_pid',
    'wrong_gpu','plan_changed','report_changed','updates','cpu','initialization','package_cpu','foreign_cuda_library',
    'numeric_source','numeric_backend','numeric_error','radius_source','radius_value','stage_exit'])
def test_falsified_cloud_engineering_cannot_release_the_gate(prepared,change):
    spec,env,ready,_,_=prepared;control=Path(env['control']);row=ready['engineering_runs'][0]
    if change=='branch_missing':ready['engineering_runs'].pop()
    elif change=='duplicate_branch':ready['engineering_runs'][1]=copy.deepcopy(row)
    elif change=='driver_unknown':row['driver']['returncode']=None
    elif change=='validator_boolean_exit':row['validation']['returncode']=False
    elif change=='same_pid':row['validation']['pid']=row['driver']['pid']
    elif change=='wrong_gpu':row['gpu']=3
    elif change=='plan_changed':Path(control/'plans'/(row['name']+'.json')).write_text('{}')
    elif change=='report_changed':Path(row['report']).write_text('{}')
    elif change in ('updates','cpu'):
        result=runtime.read(row['report'])
        if change=='updates':result['actual_updates']['actor']=29
        else:result['runtime']['device']='cpu'
        save(row['report'],result);row['report_sha256']=runtime.digest(row['report'])
    elif change=='initialization':row['initial_model_and_rng_sha256']='different'
    elif change in ('package_cpu','foreign_cuda_library'):
        path=control/'environment.json';result=runtime.read(path)
        if change=='package_cpu':result['torch']='2.12.0+cpu'
        else:result['cuda_libraries']=['/workspace/old/lib/libcublas.so.12']
        save(path,result)
    elif change.startswith('numeric_'):
        path=control/'numerical-default.json';result=runtime.read(path)
        if change=='numeric_source':result['local_source_sha256']='wrong'
        elif change=='numeric_backend':result['backend']='cpu'
        else:result['cases'][0]['max_absolute_state_error']=.1
        save(path,result)
    elif change.startswith('radius_'):
        path=control/'numerical-radius25.json';result=runtime.read(path)
        if change=='radius_source':result['checker_sha256']='wrong'
        else:result['config']['radius']=1.
        save(path,result)
    else:save(control/'processes/pip-check.json',dict(pid=1,returncode=2))
    save(spec['environment_ready'],ready)
    with pytest.raises(ValueError):runtime.inspect(spec)


@pytest.mark.parametrize('change',['old_environment','wrong_job','wrong_team','wrong_cluster','before_ready','not_saved',
    'not_usable','wrong_reference','wrong_tag','credentials','wrong_detail_endpoint','wrong_list_endpoint'])
def test_wrong_or_unusable_image_cannot_release_the_gate(prepared,change):
    spec,env,ready,image,_=prepared
    if change=='old_environment':image['environment_ready_sha256']='wrong'
    elif change=='wrong_job':image['detail']['jobId']='another-job'
    elif change=='wrong_team':image['save_request']['teamId']='another-team'
    elif change=='wrong_cluster':image['detail']['clusterId']='another-cluster'
    elif change=='before_ready':image['save_requested_at']=99.
    elif change=='not_saved':image['detail']['status']=1
    elif change=='not_usable':image['usable_record']['status']=1
    elif change=='wrong_reference':image['usable_record']['path']='registry.invalid/other:tag'
    elif change=='wrong_tag':image['detail']['tag']='other-tag'
    elif change=='credentials':image['detail']['password']='fixture-secret-never-upload'
    elif change=='wrong_detail_endpoint':image['detail_api_path']='/wrong'
    else:image['usable_api_path']='/wrong'
    save(spec['image_receipt'],image)
    with pytest.raises(ValueError):runtime.inspect(spec)


def test_load_spec_checks_all_four_files_and_the_environment_tool_identity(tmp_path):
    for name in runtime.TOOL_FILES:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('fixture source '+name)
    files={name:runtime.digest(tmp_path/name) for name in runtime.TOOL_FILES}
    env=tmp_path/'env.json';save(env,{'environment_tool_sha256':files['research/gac_environment_runtime.py']})
    path=tmp_path/'spec.json';save(path,dict(format='gac-runtime-spec-v1',controller_files=files,environment_spec=str(env),environment_spec_sha256=runtime.digest(env)))
    result=runtime.load_spec(path)
    assert result['_tools_root']==str(tmp_path) and result['_spec_sha256']==runtime.digest(path)
    (tmp_path/'research/gac_execution.py').write_text('changed')
    with pytest.raises(ValueError,match='source differs'):runtime.load_spec(path)


def test_imports_must_come_from_the_bound_release():
    runtime.check_imports({'_tools_root':str(Path(runtime.__file__).resolve().parents[1])})
    with pytest.raises(ValueError,match='outside'):runtime.check_imports({'_tools_root':'/fixture-wrong-release'})


def test_launch_requires_idle_four_card_resources_before_creating_paths(prepared,monkeypatch):
    spec,env,*_=prepared
    monkeypatch.setattr(runtime,'check_layout',lambda *_:None)
    monkeypatch.setattr(runtime.environment,'verify_artifacts',lambda *_:(object(),{'status':'artifacts_verified'}))
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda argv,**kw:'NVIDIA RTX4090\n'*4 if '--query-gpu=name' in argv else '1234\n')
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *a,**kw:pytest.fail('Occupied GPUs started scientific work'))
    assert runtime.launch(spec)['status']=='waiting_for_idle_gpus'
    assert not any(Path(spec[key]).exists() for key in ('control','results','reports'))


def test_launch_creates_one_owner_and_refuses_to_restart_its_existing_attempt(prepared,monkeypatch):
    spec,env,*_=prepared
    monkeypatch.setattr(runtime,'check_layout',lambda *_:None)
    monkeypatch.setattr(runtime.environment,'verify_artifacts',lambda *_:(object(),{'status':'artifacts_verified'}))
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda argv,**kw:'NVIDIA RTX4090\n'*4 if '--query-gpu=name' in argv else '')
    calls=[]
    class Owner:
        pid=999999999
    def spawn(*args,**kwargs):calls.append((args,kwargs));return Owner()
    monkeypatch.setattr(runtime.subprocess,'Popen',spawn)
    result=runtime.launch(spec)
    assert result['pid']==999999999 and result['runs']==32 and len(calls)==1
    assert calls[0][1]['start_new_session'] is True
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']==''
    runtime.verify_launch(spec)
    with pytest.raises(FileExistsError):runtime.launch(spec)
    assert len(calls)==1


@pytest.mark.parametrize('exit_code,result_mode,expected',[ (0,'complete',0),(2,'needs_review',2),(0,'missing',2),(7,'complete',7)])
def test_owner_observes_scheduler_exit_before_writing_terminal(prepared,monkeypatch,exit_code,result_mode,expected):
    spec,env,*_=prepared;control=Path(spec['control']);control.mkdir();(control/'tmp').mkdir()
    save(control/'launch.json',dict(spec_sha256=spec['_spec_sha256'],source_commit=runtime.execution.SOURCE,evidence={'input_hashes':{}}))
    if result_mode!='missing':save(control/'supervisor-result.json',dict(source_commit=runtime.execution.SOURCE,expected_runs=32,
        complete_runs=32 if result_mode=='complete' else 31,returncode=0 if result_mode=='complete' else 2,status=result_mode))
    class Child:
        pid=999999998
        def wait(self):
            assert not (control/'terminal.json').exists()
            return exit_code
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *a,**k:Child())
    assert runtime.own(spec)==expected
    observed=runtime.read(control/'scheduler-exit.json');terminal=runtime.read(control/'terminal.json')
    assert observed['returncode']==exit_code and terminal['scheduler_exit_observed'] is True
    if exit_code!=0 or result_mode=='missing':assert terminal['status']=='needs_review'
    with pytest.raises(FileExistsError):runtime.own(spec)


def test_changed_launch_evidence_is_rejected(prepared):
    spec,env,*_=prepared;control=Path(spec['control']);control.mkdir()
    path=Path(spec['image_receipt'])
    save(control/'launch.json',dict(spec_sha256=spec['_spec_sha256'],source_commit=runtime.execution.SOURCE,
         evidence={'input_hashes':{str(path):runtime.digest(path)}}))
    path.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):runtime.verify_launch(spec)


def test_live_driver_prevents_terminal_analysis(prepared,monkeypatch):
    spec,env,_,_,rows=prepared;control=Path(spec['control']);row=rows[0]
    save(control/'runs'/(row['name']+'.json'),dict(name=row['name'],seed=row['seed'],pid=1234,status='running'))
    monkeypatch.setattr(runtime,'active_driver',lambda *_:True)
    with pytest.raises(RuntimeError,match='still live'):runtime.collect_records(spec,rows,Path(env['runtime_root'])/'source')


def test_complete_receipt_cannot_change_metrics_while_keeping_the_old_validation_hash(prepared):
    spec,env,_,_,rows=prepared;control=Path(spec['control']);row=rows[0]
    result=dict(status='complete',returncode=0,metrics={'auc':.1})
    validated=control/'validations'/(row['name']+'.json');save(validated,result)
    save(control/'runs'/(row['name']+'.json'),dict(result,name=row['name'],seed=row['seed'],
         metrics={'auc':.9},validation_report_sha256=runtime.digest(validated)))
    with pytest.raises(ValueError,match='validation output changed'):runtime.collect_records(spec,rows,Path(env['runtime_root'])/'source')


def test_image_api_projection_discards_credentials_and_other_unneeded_fields():
    raw=dict(id='fixture',path='registry.invalid/team/image:tag',status=2,username='private-user',
             password='fixture-secret',content='unneeded-description',token='fixture-token')
    assert runtime.image_projection(raw)==dict(id='fixture',path='registry.invalid/team/image:tag',status=2)
    assert raw['password']=='fixture-secret'


def test_production_layout_cannot_redirect_scientific_outputs(prepared):
    spec,env,*_=prepared
    with pytest.raises(ValueError,match='layout'):runtime.check_layout(spec,env)


@pytest.mark.parametrize('failed',[False,True])
def test_supervisor_collects_all_registered_receipts_before_all_twenty_four_contrasts(prepared,monkeypatch,failed):
    spec,env,_,_,rows=prepared;control=Path(spec['control']);control.mkdir();(control/'tmp').mkdir()
    save(control/'launch.json',dict(spec_sha256=spec['_spec_sha256'],source_commit=runtime.execution.SOURCE,
         evidence={'input_hashes':{},'image':{'fixture_only':True}}))
    Path(spec['reports']).mkdir()
    bundle,manifest,cohort,_=runtime.verified_runtime(spec,env)
    manifest['files']['analysis/research/gac_reference_validation.py']='d'*64
    fixtures=design.fixture_reports(rows)[runtime.GAC]['records']
    indexed={(r['name'],r['seed']):r for r in fixtures}
    calls=[]
    def execute(row,**kwargs):
        record=copy.deepcopy(indexed[row['name'],row['seed']]);calls.append(row['name'])
        if failed and row['name']==rows[0]['name']:
            record.update(status='failed',returncode=7)
        else:
            path=control/'validations'/(row['name']+'.json');save(path,record)
            record['validation_report_sha256']=runtime.digest(path)
        save(control/'runs'/(row['name']+'.json'),record)
        return record
    monkeypatch.setattr(runtime.execution,'execute_case',execute)
    assert runtime.supervise(spec)==(2 if failed else 0)
    assert len(calls)==32 and not (control/'terminal.json').exists() # Only the observing owner writes terminal.
    report=runtime.read(Path(spec['reports'])/'gac-run-report.json')
    comparison=runtime.read(Path(spec['reports'])/'gac-comparison-report.json')
    assert report['complete_runs']==(31 if failed else 32) and len(comparison['comparisons'])==24
    if failed:
        incomplete=[r for r in comparison['comparisons'] if r['status']=='incomplete']
        assert incomplete and all('summary' not in r for r in incomplete)
    else:assert comparison['status']=='complete'
    with pytest.raises(FileExistsError):runtime.supervise(spec)
