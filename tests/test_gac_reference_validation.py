import copy
import json

import numpy as np
import pytest
import torch

from research.gac_reference_validation import (
    assert_checkpoint_equal, curve_metrics, expected_grid, expected_updates,
    validate_actions, validate_run, validate_training,
)


def plan(**overrides):
    result = dict(online_steps=100, eval_interval=50, eval_episodes=2, eval_seed=70, log_interval=32,
                  action_execution='clip', learner=dict(radius=2.5, learning_starts=32,
                  policy_frequency=2, target_network_frequency=1))
    result.update(overrides)
    return result


def curve():
    rows = []
    for step, count in [(0, 0), (50, 1), (100, 2)]:
        episodes = [dict(episode=i, reset_seed=70+i, length=3, success=int(i<count), clipped_action_steps=1)
                    for i in range(2)]
        rows.append(dict(step=step, episode_count=2, success_count=count, success=count/2, episodes=episodes))
    return rows


def test_auc_uses_real_step_zero_and_the_entire_online_budget():
    result = curve_metrics(curve(), plan())
    assert result['auc'] == .5 and result['endpoint'] == 1 and result['tail3'] == .5
    assert result['time_to_50'] == 50 and not result['time_to_50_censored']
    assert result['evaluation_clipped_steps'] == 6 and result['tail_evaluations'] == 3


@pytest.mark.parametrize('change', ['missing_zero', 'duplicate', 'order', 'count', 'seed', 'aggregate', 'clipped', 'identity'])
def test_forged_or_incomplete_evaluation_evidence_is_rejected(change):
    rows, p = curve(), plan()
    if change == 'missing_zero': rows.pop(0)
    elif change == 'duplicate': rows.append(copy.deepcopy(rows[-1]))
    elif change == 'order': rows[0], rows[1] = rows[1], rows[0]
    elif change == 'count': rows[0]['episode_count'] = 50
    elif change == 'seed': rows[0]['episodes'][0]['reset_seed'] = 99
    elif change == 'aggregate': rows[0]['success'] = .5
    elif change == 'clipped': rows[0]['episodes'][0]['clipped_action_steps'] = 4
    else: p['action_execution'] = 'identity'
    with pytest.raises(ValueError): curve_metrics(rows, p)


def test_all_zero_and_nonmultiple_endpoint_preserve_censoring():
    p = plan(online_steps=75, eval_episodes=1)
    assert expected_grid(p) == [0, 50, 75]
    rows = [dict(step=step, episode_count=1, success_count=0, success=0,
                 episodes=[dict(episode=0, reset_seed=70, length=2, success=0, clipped_action_steps=0)])
            for step in expected_grid(p)]
    result = curve_metrics(rows, p)
    assert result['auc'] == 0 and result['time_to_50'] is None and result['time_to_50_censored']


def action_rows():
    return [dict(step=i+1, zero_based_loop=i, warmup=i==0, terminated=False, truncated=False,
                 requested_action=[.9]*5 if i==0 else [1.5, 0, 0, 0, 0],
                 action=[.9]*5 if i==0 else [1, 0, 0, 0, 0]) for i in range(3)]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_action_audit_preserves_requested_prefix_and_accepts_box_warmup(tmp_path):
    p = plan(online_steps=3, learner=dict(radius=2.5, learning_starts=1))
    path = tmp_path/'actions.jsonl'; write_rows(path, action_rows())
    result = validate_actions(path, p, prefix_limit=2)
    assert result['execution_clipped_steps'] == 2 and result['max_requested_abs'] == 1.5
    assert result['max_executed_abs'] == 1 and result['prefix_requested_actions'][1][0] == 1.5
    rows = action_rows(); rows[1]['requested_action'] = rows[1]['action'] = [0]*5
    rows[2]['requested_action'] = rows[2]['action'] = [0]*5
    p['action_execution'] = 'identity'; p['learner']['radius'] = 1
    write_rows(path, rows)
    assert validate_actions(path, p)['execution_clipped_steps'] == 0


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'warmup', 'termination', 'map', 'radius', 'nan', 'shape'])
def test_action_failures_are_not_hidden_by_clipping(tmp_path, change):
    rows = action_rows()
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    elif change == 'warmup': rows[1]['warmup'] = True
    elif change == 'termination': rows[1]['terminated'] = 1
    elif change == 'map': rows[1]['action'][0] = .5
    elif change == 'radius': rows[1]['requested_action'][0] = 3
    elif change == 'nan': rows[1]['requested_action'][0] = float('nan')
    else: rows[1]['action'].pop()
    path = tmp_path/'actions.jsonl'; write_rows(path, rows)
    with pytest.raises(ValueError): validate_actions(path, plan(online_steps=3, learner=dict(radius=2.5, learning_starts=1)))


@pytest.mark.parametrize('budget,start,actor,target', [(64,32,2,1), (5,5,2,1), (100,3,3,4), (10,0,7,3)])
def test_update_counts_match_zero_based_event_enumeration(budget, start, actor, target):
    p = plan(online_steps=budget, learner=dict(learning_starts=start, policy_frequency=actor, target_network_frequency=target))
    events = [step for step in range(budget) if step > start]
    assert expected_updates(p) == dict(critic=len(events), actor=sum(actor for step in events if step%actor==0),
                                        target=sum(step%target==0 for step in events))


def test_million_step_counts_do_not_round_the_warmup_boundary():
    p = plan(online_steps=1_000_000, learner=dict(learning_starts=5000, policy_frequency=2, target_network_frequency=1))
    assert expected_updates(p) == dict(critic=994999, actor=994998, target=994999)


def training_row():
    return dict(step=64, global_step=63, actor_global_step=62, actor_loss=0., sac_actor_loss=0.,
                kappa_reg_loss=0., qf_loss=0., qf1_loss=0., qf2_loss=0.)


@pytest.mark.parametrize('change', ['valid', 'missing', 'actor_timestamp', 'clock', 'nonfinite', 'missing_loss'])
def test_training_diagnostics_require_the_actual_actor_event(change):
    rows = [training_row()]
    if change == 'missing': rows = []
    elif change == 'actor_timestamp': rows[0]['actor_global_step'] = 63
    elif change == 'clock': rows[0]['global_step'] = 64
    elif change == 'nonfinite': rows[0]['qf_loss'] = float('nan')
    elif change == 'missing_loss': rows[0].pop('actor_loss')
    if change == 'valid': assert validate_training(rows, plan(online_steps=64)) == 1
    else:
        with pytest.raises(ValueError): validate_training(rows, plan(online_steps=64))


def test_no_update_budget_has_no_fabricated_diagnostics():
    assert validate_training([], plan(online_steps=32)) == 0


@pytest.mark.parametrize('change', ['valid', 'dtype', 'nonfinite', 'keys', 'value'])
def test_checkpoint_comparison_checks_values_dtypes_and_finiteness(change):
    left = dict(models={'a':torch.tensor([1.])}, config={'radius':2.5}, clock=None)
    right = copy.deepcopy(left)
    if change == 'dtype': right['models']['a'] = right['models']['a'].double()
    elif change == 'nonfinite': left['models']['a'][0] = right['models']['a'][0] = float('nan')
    elif change == 'keys': right.pop('clock')
    elif change == 'value': right['config']['radius'] = 1.
    if change == 'valid': assert assert_checkpoint_equal(left, right) == 3
    else:
        with pytest.raises(ValueError): assert_checkpoint_equal(left, right)


@pytest.mark.parametrize('exit_code', [None, 1, False])
def test_receipts_cannot_replace_an_observed_successful_exit(tmp_path, exit_code):
    with pytest.raises(ValueError, match='lifecycle'):
        validate_run(tmp_path, tmp_path/'missing-plan.json', dict(returncode=exit_code), tmp_path)
