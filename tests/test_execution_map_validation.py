"""Reject evidence that cannot establish actual matched-map initialization."""
from copy import deepcopy

import pytest

from research import execution_map_validation as v


def records():
    return [dict(task=task, execution_map=mapping, seed=9824, status='complete',
        source_commit=v.common.SOURCE, driver_exit_observed=True,
        driver_returncode=0, validation_returncode=0, outcomes_inspected=False,
        initial_receipt_kind='actual_constructor_capture_before_initial_evaluation',
        actual_initial_serialized_agent_sha256=('a' if task == 'triple2' else 'b')*64,
        warmup_actions_sha256=('c' if task == 'triple2' else 'd')*64)
        for task in v.TASKS for mapping in v.MAPS]


def test_all_six_observed_exits_and_actual_initializations_are_required():
    result = v.validate_groups(records())
    assert result['status'] == 'all_six_actual_initializations_verified'
    assert result['simulator_state_matched'] is False
    assert result['initial_execution_distribution_matched'] is False
    assert result['cloud_4090_validation'] is False


@pytest.mark.parametrize('field,value', [
    ('driver_returncode', None), ('driver_returncode', False), ('driver_returncode', 1),
    ('driver_exit_observed', False), ('validation_returncode', None),
    ('validation_returncode', 1), ('status', 'running'), ('outcomes_inspected', True),
    ('initial_receipt_kind', 'reconstructed_from_flags'),
    ('actual_initial_serialized_agent_sha256', 'different-state'),
    ('warmup_actions_sha256', 'different-commands'), ('seed', 9825),
    ('source_commit', 'another-release'),
])
def test_incomplete_or_unmatched_evidence_cannot_be_promoted(field, value):
    rr = records(); rr[1][field] = value
    with pytest.raises(ValueError):
        v.validate_groups(rr)


def test_duplicate_case_does_not_replace_a_missing_map():
    rr = records(); rr[-1] = deepcopy(rr[-2])
    with pytest.raises(ValueError, match='Missing/duplicate'):
        v.validate_groups(rr)


def test_partial_matrix_does_not_establish_two_task_acceptance():
    with pytest.raises(ValueError, match='Missing/duplicate'):
        v.validate_groups(records()[:3])


def identity(mapping):
    from agents.acrlpd import get_config
    from direction_speed.config import configure_direction_speed
    config = get_config().to_dict()
    config.update(horizon_length=1, action_dim=6, env_action_dim=5,
        policy_action_dim=6, critic_action_dim=5, target_entropy=-3.)
    configure_direction_speed(ds_mode='adapter', ds_speed_bound='sphere',
        ds_adapter_representation='cartesian', ds_adapter_speed_squash='tanh',
        ds_adapter_speed_interpretation='log', ds_adapter_critic_space='env',
        ds_adapter_execution_map=mapping, config=config, agent_name='acrlpd')
    config['actor_hidden_dims'] = list(config['actor_hidden_dims'])
    config['value_hidden_dims'] = list(config['value_hidden_dims'])
    case = dict(index=0, task='triple2', method='full', execution_map=mapping,
        seed=9824, run_group='capture', argv=['python', 'main_online.py',
            '--agent=agents/acrlpd.py', '--seed=9824',
            '--env_name='+v.TASKS['triple2'], '--run_group=capture',
            '--ds_adapter_execution_map='+mapping, '--ds_adapter_critic_space=env',
            '--agent.adapter_actor_q_gradient=full'])
    receipt = dict(source_commit=v.common.SOURCE, source_binding_sha256='source-hash',
        seed=9824, method='full', case_index=0, network_step=1, backend='gpu',
        device_kind='NVIDIA RTX 6000 Ada', serialized_agent_sha256='a'*64,
        training_flags=case['argv'][2:])
    return case, receipt, config


@pytest.mark.parametrize('mapping', v.MAPS)
def test_real_resolved_config_and_bound_constructor_identity(mapping):
    case, receipt, config = identity(mapping)
    v.validate_identity(case, receipt, config,
        source_binding_sha='source-hash', expected_device='RTX 6000 Ada')


@pytest.mark.parametrize('field,value', [
    ('source_binding_sha256', 'changed'), ('case_index', 1), ('seed', 0),
    ('method', 'remove_direction_radial'), ('network_step', 34),
    ('backend', 'cpu'), ('device_kind', 'different GPU'),
    ('serialized_agent_sha256', 'not-a-digest'), ('training_flags', []),
])
def test_mismatched_actual_constructor_is_rejected(field, value):
    case, receipt, config = identity('cartesian_projection_aux')
    receipt[field] = value
    with pytest.raises(ValueError, match='constructor'):
        v.validate_identity(case, receipt, config,
            source_binding_sha='source-hash', expected_device='RTX 6000 Ada')


@pytest.mark.parametrize('field,value', [('critic_action_dim', 6),
    ('target_entropy', -2.5), ('adapter_critic_space', 'latent'),
    ('adapter_actor_q_gradient', 'remove_direction_radial')])
def test_capacity_regularizer_and_critic_recipe_changes_are_rejected(field, value):
    case, receipt, config = identity('cartesian_aux'); config[field] = value
    with pytest.raises(ValueError, match='recipe'):
        v.validate_identity(case, receipt, config,
            source_binding_sha='source-hash', expected_device='RTX 6000 Ada')
