import copy
import json
from pathlib import Path

import pytest

import research.gac_environment_runtime as runtime


def completed_fql():
    terminal = dict(status='complete', source_commit=runtime.FQL_SOURCE, expected_runs=16, complete_runs=16)
    report = dict(terminal, records=[], comparisons=[])
    for env in runtime.TASKS:
        for method in ('native', 'posthoc'):
            for seed in runtime.SEEDS:
                report['records'].append(dict(env_name=env, method=method, seed=seed, status='complete',
                    source_commit=runtime.FQL_SOURCE, returncode=0, validation_returncode=0,
                    offline_steps=1_000_000, online_steps=1_000_000, actual_joint_optimizer_updates=2_000_000,
                    actual_environment_steps=1_000_000, observed_evaluation_episode_counts='verified', runtime={'backend':'gpu'}))
    for task in ('triple2', 'double2'):
        report['comparisons'].append(dict(task=task, status='complete', unavailable_seeds=[],
            pairs=[dict(seed=seed) for seed in runtime.SEEDS],
            summary={key:{'mean':-.25} for key in ('online_auc','endpoint','online_tail3','offline_auc','offline_endpoint')}))
    return terminal, report, dict(returncode=0, lane_errors=[])


def test_negative_outcomes_do_not_prevent_the_technical_gate():
    result = runtime.verify_fql_gate(*completed_fql())
    assert result['runs'] == 16 and result['performance_threshold'] is None and not result['scientific_selection']


@pytest.mark.parametrize('change', ['terminal_source','terminal_count','status','duplicate','missing','budget','updates',
    'observed_count','cpu','exit_unknown','exit_boolean','validation_missing','validation_boolean','pair_missing',
    'comparison_missing','summary_missing','scheduler_unknown','scheduler_failed','lane_error'])
def test_incomplete_or_mismatched_upstream_cannot_release_the_gate(change):
    terminal, report, scheduler = completed_fql()
    if change=='terminal_source': terminal['source_commit']='wrong-source'
    elif change=='terminal_count': terminal['complete_runs']=15
    elif change=='status': report['records'][0]['status']='failed'
    elif change=='duplicate': report['records'][1]=copy.deepcopy(report['records'][0])
    elif change=='missing': report['records'].pop()
    elif change=='budget': report['records'][0]['online_steps']=999999
    elif change=='updates': report['records'][0]['actual_joint_optimizer_updates']=1999999
    elif change=='observed_count': report['records'][0]['observed_evaluation_episode_counts']='flags_only'
    elif change=='cpu': report['records'][0]['runtime']['backend']='cpu'
    elif change=='exit_unknown': report['records'][0]['returncode']=None
    elif change=='exit_boolean': report['records'][0]['returncode']=False
    elif change=='validation_missing': report['records'][0].pop('validation_returncode')
    elif change=='validation_boolean': report['records'][0]['validation_returncode']=False
    elif change=='pair_missing': report['comparisons'][0]['pairs'].pop()
    elif change=='comparison_missing': report['comparisons'].pop()
    elif change=='summary_missing': report['comparisons'][0].pop('summary')
    elif change=='scheduler_unknown': scheduler['returncode']=None
    elif change=='scheduler_failed': scheduler['returncode']=2
    else: scheduler['lane_errors']=[{'error':'fixture'}]
    with pytest.raises(ValueError): runtime.verify_fql_gate(terminal,report,scheduler)


def test_missing_upstream_is_read_only_and_prepare_cannot_create_prefix(tmp_path,monkeypatch):
    spec={key:str(tmp_path/(key+'.json')) for key in ('gate_terminal','gate_report','gate_scheduler_exit')}
    control,prefix=tmp_path/'control',tmp_path/'prefix';spec.update(control=str(control),prefix=str(prefix))
    monkeypatch.setattr(runtime,'verify_artifacts',lambda *_:pytest.fail('Artifact/install phase ran before upstream completion'))
    assert runtime.inspect(spec)['status']=='waiting_for_fql'
    with pytest.raises(RuntimeError,match='FQL is not technically complete'):runtime.prepare(spec)
    assert not control.exists() and not prefix.exists() and list(tmp_path.iterdir())==[]


def test_completed_upstream_still_cannot_install_an_incomplete_archive(tmp_path,monkeypatch):
    spec={key:str(tmp_path/(key+'.json')) for key in ('gate_terminal','gate_report','gate_scheduler_exit')}
    for key,value in zip(('gate_terminal','gate_report','gate_scheduler_exit'),completed_fql()):
        Path(spec[key]).write_text(json.dumps(value))
    monkeypatch.setattr(runtime,'verify_artifacts',lambda *_:(None,{'status':'waiting_for_dependency_transfer'}))
    assert runtime.inspect(spec)['status']=='waiting_for_dependency_transfer'
    with pytest.raises(RuntimeError,match='Dependency transfer is incomplete'):runtime.prepare(spec)
    assert not (tmp_path/'control').exists()


def test_environment_drops_old_python_and_workspace_libraries_but_retains_driver_paths():
    value=runtime.clean_environment('/opt/fixture-prefix','/workspace/fixture-temp','2',dict(
        PYTHONPATH='/workspace/venv/lib',PYTHONHOME='/workspace/venv',VIRTUAL_ENV='/workspace/venv',
        PATH='/workspace/venv/bin:/usr/bin',LD_LIBRARY_PATH='/workspace/venv/lib:/usr/local/nvidia/lib64'))
    assert not {'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}&set(value)
    assert value['PATH']=='/opt/fixture-prefix/bin:/usr/bin'
    assert value['LD_LIBRARY_PATH']=='/usr/local/nvidia/lib64'
    assert value['CUDA_VISIBLE_DEVICES']=='2' and value['PYTHONNOUSERSITE']=='1' and value['PIP_NO_CACHE_DIR']=='1'


def test_mount_matching_does_not_confuse_a_pvc_with_the_image_root():
    lines='1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / /workspace rw - dpc source rw\n'
    assert runtime.mount_for('/opt/environment',lines)==('/','overlay')
    assert runtime.mount_for('/workspace/output',lines)==('/workspace','dpc')
    assert runtime.mount_for('/workspace-other/output',lines)==('/','overlay')


def test_cuda_library_check_accepts_only_the_new_prefix_for_user_libraries():
    prefix='/opt/fixture-prefix'
    paths=['/opt/fixture-prefix/lib/libcublas.so.12','/opt/fixture-prefix/lib/libcudart.so.12','/usr/local/nvidia/lib64/libcuda.so.1']
    maps='\n'.join('range r-xp offset device inode '+p for p in paths)
    assert runtime.cuda_library_paths(prefix,maps)==sorted(paths[:2])
    with pytest.raises(ValueError):runtime.cuda_library_paths(prefix,maps+'\nrange r-xp offset device inode /workspace/venv/lib/libcublas.so.12')
    with pytest.raises(ValueError,match='No loaded CUDA'):runtime.cuda_library_paths(prefix,'range r-xp offset device inode /usr/local/nvidia/lib64/libcuda.so.1')


def test_engineering_templates_cover_all_eight_branches_without_mutating_science(tmp_path):
    science=tmp_path/'science';science.mkdir();(science/'plans').mkdir()
    cohort=dict(seeds=list(runtime.SEEDS),replay_protocols=['online_only','prior_balanced'],profiles=[{'name':'r1'},{'name':'r25'}])
    rows=[];originals={}
    for env_index,env in enumerate(runtime.TASKS):
        for profile in ('r1','r25'):
            for protocol in ('online_only','prior_balanced'):
                name=f'fixture{env_index}_{profile}_{protocol}';relative='plans/'+name+'.json'
                plan=dict(seed=10,env_name=env,online_steps=1_000_000,eval_episodes=50,replay_capacity=1_000_000,
                          learner={'learning_starts':5000,'radius':1. if profile=='r1' else 2.5},replay_protocol=protocol)
                (science/relative).write_text(json.dumps(plan));originals[relative]=(science/relative).read_bytes()
                rows.append(dict(seed=10,env_name=env,task=f'fixture{env_index}',profile=profile,protocol=protocol,plan=relative))
    (science/'cohort.json').write_text(json.dumps(cohort));(science/'tasks.json').write_text(json.dumps(rows))
    result=runtime.engineering_plans(tmp_path)
    assert len(result)==8 and {r['gpu'] for r in result}=={0,1,2,3}
    assert all(r['plan']['online_steps']==64 and r['plan']['eval_episodes']==1 and r['plan']['learner']['learning_starts']==32 for r in result)
    assert {r['plan']['seed'] for r in result}=={9601,9602}
    assert all((science/path).read_bytes()==data for path,data in originals.items())


@pytest.mark.parametrize('program',[runtime.CHECK_ENV,runtime.VALIDATE_CASE,runtime.RADIUS_ORACLE])
def test_exported_subprocess_programs_compile(program):
    compile(program,'exported_gac_environment_program','exec')
