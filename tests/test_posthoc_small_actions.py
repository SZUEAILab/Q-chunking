import numpy as np
import pytest
from scipy.special import ndtr

from research.posthoc_small_actions import gaussian_box_log_bounds, small_action_bounds, logmeanexp, event_summary


def test_density_bounds_bracket_independent_gaussian_box_probability():
    loc = np.array([[0., .2, -1.], [.3, -.7, 1.2]])
    std = np.array([[1., .7, .4], [.6, .8, 1.3]])
    for width in [.1, .4, 1.]:
        lo, hi = gaussian_box_log_bounds(loc, std, width)
        exact = np.log(ndtr((width-loc)/std) - ndtr((-width-loc)/std)).sum(axis=-1)
        assert np.all(lo <= exact + 1e-13) and np.all(exact <= hi + 1e-13)


def test_centered_gaussian_tiny_box_upper_bound_and_cubic_epsilon_dependence():
    first = small_action_bounds(np.zeros((1, 3)), np.ones((1, 3)), .3)
    width = np.arctanh(1e-6 * .3 * np.e)
    expected = 3 * np.log10(2 * width / np.sqrt(2 * np.pi))
    assert first['log10_upper_posthoc_probability_mean'] == pytest.approx(expected)
    second = small_action_bounds(np.zeros((1, 3)), np.ones((1, 3)), .3, eps=1e-3)
    assert second['log10_upper_posthoc_probability_mean'] - first['log10_upper_posthoc_probability_mean'] == pytest.approx(9., abs=1e-6)


def test_probability_average_uses_logsumexp_not_mean_log_probability():
    assert logmeanexp(np.array([-1000., -1000.])) == -1000.
    assert logmeanexp(np.log([1e-20, 1e-30])) == pytest.approx(np.log((1e-20+1e-30)/2))
    bounds = small_action_bounds(np.array([[0., 0., 0.], [1., 1., 1.]]), np.ones((2, 3)), .3)
    per_state = np.array(bounds['log10_upper_posthoc_probability_per_state'])
    assert bounds['log10_upper_posthoc_probability_mean'] == pytest.approx(np.log10(np.mean(10**per_state)))


@pytest.mark.parametrize('threshold,eps', [(0., 1e-6), (.5, 1e-6), (.1, 0.), (.1, np.nan)])
def test_bound_domain_rejects_regular_branch_thresholds_or_invalid_floor(threshold, eps):
    with pytest.raises(ValueError): small_action_bounds(np.zeros((2, 3)), np.ones((2, 3)), threshold, eps)


def test_invalid_gaussian_and_nonfinite_draws_are_rejected():
    with pytest.raises(ValueError): gaussian_box_log_bounds(np.zeros((1, 3)), np.zeros((1, 3)), .1)
    with pytest.raises(ValueError): gaussian_box_log_bounds(np.zeros((1, 2)), np.ones((1, 2)), .1)
    with pytest.raises(ValueError): event_summary(np.full((2, 3, 5), np.nan), [.3])


def test_actual_legacy_mapper_regular_branch_floor_and_small_action_reachability():
    from direction_speed.adapter import DirectionSpeedAdapter
    mapper = DirectionSpeedAdapter(speed_squash='tanh', speed_interpretation='log', speed_bound='sphere')
    rng = np.random.default_rng(20261025)
    directions = rng.normal(size=(2000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    direction_norms = np.geomspace(mapper.eps * 1.001, 1., 2000)
    latent = np.c_[directions * direction_norms[:, None], np.linspace(-1., 1., 2000), np.zeros((2000, 2))]
    radii = np.linalg.norm(mapper.compose(latent)[:, :3], axis=-1)
    assert np.all(radii >= np.exp(-1) - 1e-12)
    desired = np.array([.05, -.02, .03, .4, -.3])
    small = np.r_[mapper.eps * desired[:3], 0., desired[3:]]
    np.testing.assert_allclose(mapper.compose(small[None])[0], desired, atol=1e-15, rtol=1e-15)


def test_same_head_raw_small_box_bounds_descriptive_events_without_zero_inference():
    actions = np.zeros((2, 3, 5));actions[0, 1:, 0] = .5;actions[1, 1:, 0] = .2
    small = event_summary(actions, [.1, .3])
    assert [row['count'] for row in small] == [2, 4]
    assert small[1]['per_state_fraction'] == [1/3, 1.]
