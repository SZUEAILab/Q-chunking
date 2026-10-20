import copy
import pytest
from research.actor_radial_summary import FIELDS, summarize, write_summary


@pytest.fixture
def inputs():
    plan=dict(training_source_commit='source',tool_sha256='tool',expected_cases=4,actor_noise_seed=23,
              cases=[],panels=dict(triple2=dict(panel='fixed')))
    report=dict(training_source_commit='source',tool_sha256='tool',expected_cases=4,records=[])
    for index,seed in enumerate([10,11,12,13]):
        case=dict(index=index,task='triple2',method='legacy_latent',seed=seed,run_dir='run-'+str(seed),
                  checkpoint_sha256='checkpoint',config_sha256='config',flags_sha256='flags',network_step=9)
        metrics=dict(samples=128,seed=23,valid_direction_samples=128,small_direction_samples=0,
            optimizer_updates_performed=0,network_step_before=9,network_step_after=9,
            **dict.fromkeys(FIELDS,float(index+1)))
        plan['cases'].append(case);report['records'].append(dict(case,status='complete',returncode=0,
            source_commit='source',tool_sha256='tool',panel=plan['panels']['triple2'],metrics=metrics))
    return report,plan


def test_four_seed_mean_and_undefined_unit_response_are_not_zero_or_partial_means(inputs,tmp_path):
    report,plan=inputs;report['records'][0]['metrics']['unit_radial_parameter_environment_response_rms']=None
    result=summarize(report,plan);row=result['groups'][0]
    assert row['summary']['radial_q_actor_gradient_norm']['mean']==2.5
    assert row['summary']['unit_radial_parameter_environment_response_rms']['mean'] is None
    assert row['summary']['unit_radial_parameter_environment_response_rms']['defined_seeds']==3
    write_summary(tmp_path/'summary.json',result)
    assert 'N/A (3/4 defined)' in (tmp_path/'summary.md').read_text()
    with pytest.raises(FileExistsError):write_summary(tmp_path/'summary.json',result)


def test_failed_seed_is_retained_and_prevents_any_group_mean(inputs):
    report,plan=inputs;report['records'][2]['status']='needs_review';report['records'][2]['returncode']=2
    result=summarize(report,plan);row=result['groups'][0]
    assert result['status']=='incomplete' and result['available_cases']==3
    assert len(row['seeds'])==4 and row['seeds'][2]['seed']==12
    assert all(v['mean'] is None for v in row['summary'].values())


@pytest.mark.parametrize('field,value',[('checkpoint_sha256','other'),('seed',99),('source_commit','other'),('returncode',False)])
def test_changed_complete_case_is_rejected(inputs,field,value):
    report,plan=inputs;report['records'][0][field]=value
    with pytest.raises(ValueError):summarize(report,plan)


def test_duplicate_identity_is_rejected(inputs):
    report,plan=inputs;report['records'][1]['index']=0
    with pytest.raises(ValueError):summarize(report,plan)


def test_large_norm_ratio_is_preserved_as_ratio_not_clamped_to_percentage(inputs):
    report,plan=inputs
    for row in report['records']:row['metrics']['radial_to_total_q_actor_norm']=2.
    result=summarize(report,plan)
    assert result['groups'][0]['summary']['radial_to_total_q_actor_norm']['mean']==2.
