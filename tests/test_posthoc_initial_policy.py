import numpy as np
import pytest
from research.posthoc_initial_policy import conditional_moments,compare_forward_maps
from direction_speed.adapter import DirectionSpeedAdapter


def test_between_state_variation_is_not_reported_as_noise():
    a=np.zeros((3,8,5));a[:,:,0]=np.array([.1,.4,.8])[:,None]
    r=conditional_moments(a)
    assert r['mean_within_state_radius_std']<1e-15
    assert r['mean_within_state_xyz_covariance_trace']<1e-15
    assert r['between_state_radius_mean_std']>.2 and r['pooled_xyz_covariance_trace']>.05


def test_mapping_comparison_preserves_scalar_heads_and_counts_clipping():
    loc=np.zeros((3,6));std=np.ones((3,6));noise=np.random.default_rng(1).normal(size=(3,128,6))
    a=DirectionSpeedAdapter(speed_squash='tanh',speed_interpretation='log',speed_bound='sphere')
    r=compare_forward_maps(loc,std,noise,a.compose,noise_scale=.2)
    assert r['scalar_execution_max_difference']==0
    assert r['mean_radius_difference']>.5 and r['actual_xyz_clipping_fraction']>0
    assert r['normalization_floor_fraction']==0


def test_state_dependent_speed_location_is_not_replaced_with_zero():
    loc=np.zeros((2,6));loc[:,3]=[-2,2];std=np.full((2,6),.1)
    noise=np.random.default_rng(2).normal(size=(2,64,6));a=DirectionSpeedAdapter(speed_squash='tanh',speed_interpretation='log',speed_bound='sphere')
    r=compare_forward_maps(loc,std,noise,a.compose)
    assert r['posthoc']['radius_per_state_mean'][0]<.5
    assert r['posthoc']['radius_per_state_mean'][1]>1.3
    assert r['raw_speed_location_per_state']==[-2,2]


@pytest.mark.parametrize('shape',[(1,8,5),(2,1,5),(2,8,4)])
def test_insufficient_or_wrong_shaped_samples_rejected(shape):
    with pytest.raises(ValueError):conditional_moments(np.zeros(shape))
