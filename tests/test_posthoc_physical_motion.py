import copy
import numpy as np
import pytest

from research.posthoc_physical_motion import motion_measurements, assert_same_start, summarize


def measure(action, target, after):
    return motion_measurements(np.zeros(3), target, after, action,
        np.array([-.05, -.05, -.05, -.3, -1]), np.array([.05, .05, .05, .3, 1]))


def test_requested_target_and_physical_motion_have_distinct_units_and_values():
    row = measure([.4, 0, 0, .2, .1], [.02, 0, 0], [.005, 0, 0])
    assert row['action_xyz_norm'] == .4
    assert row['requested_norm_m'] == pytest.approx(.02)
    assert row['target_norm_m'] == .02 and row['actual_norm_m'] == .005
    assert not row['workspace_clipped']


def test_workspace_clipping_is_not_a_lower_bound_on_physical_motion():
    row = measure([1, 0, 0, 0, 0], [.01, 0, 0], [.002, 0, 0])
    assert row['workspace_clipped'] and row['requested_norm_m'] == .05
    assert row['actual_norm_m'] < row['target_norm_m'] < row['requested_norm_m']


def test_float32_action_rounding_is_not_misclassified_as_workspace_clipping():
    action = np.array([.237, -.113, .321, .1, .2], dtype='float32')
    low, high = -np.array([.05, .05, .05, .3, 1.]), np.array([.05, .05, .05, .3, 1.])
    request = .5 * (action + 1) * (high - low) + low
    before = np.array([.4, 0, .2])
    row = motion_measurements(before, before+request[:3], before+.1*request[:3], action, low, high)
    assert not row['workspace_clipped']


@pytest.mark.parametrize('action', [[2, 0, 0, 0, 0], [np.nan, 0, 0, 0, 0], [0, 0, 0]])
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError): measure(action, [0, 0, 0], [0, 0, 0])


def test_matching_observations_alone_do_not_certify_matching_full_states():
    assert_same_start(np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(AssertionError): assert_same_start(np.zeros(2), np.array([0, 0, 1e-15]), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError): assert_same_start(np.zeros(2), np.array([np.nan]), np.zeros(2), np.array([np.nan]))


def records():
    result = []
    for arm, displacement in [('native', .01), ('posthoc', .005), ('same_head_cartesian', .003), ('zero_xyz', .001)]:
        result.append(dict(seed=10, noise_scale=1., reset_seed=100, draw=0, arm=arm, start_state_sha256='same',
            action_xyz_norm=.5, requested_norm_m=.025, target_norm_m=.02, actual_norm_m=displacement,
            actual_displacement_m=[displacement, 0, 0], workspace_clipped=False))
    return result


def test_zero_xyz_correction_uses_paired_vectors_before_norms():
    output = summarize(records(), [10], [1.], [100], 1)[0]
    assert output['mean_zero_xyz_corrected_norm_m'] == pytest.approx(dict(posthoc=.004, same_head_cartesian=.002))
    assert output['paired_corrected_norm_difference_m'] == pytest.approx(.002)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'state'])
def test_incomplete_duplicate_or_unmatched_pairs_do_not_produce_aggregates(problem):
    rows = records()
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    else: rows[-1]['start_state_sha256'] = 'different'
    with pytest.raises(ValueError): summarize(rows, [10], [1.], [100], 1)
