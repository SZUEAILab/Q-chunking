import numpy as np
import pytest
from research.posthoc_mechanism import probe,replay_geometry,jacobian_check,normalization_boundary,legacy_adapter


def test_regular_branch_floor_and_subepsilon_exception():
    rng=np.random.default_rng(2);p=np.tanh(rng.normal(size=(1000,6)))
    r=np.linalg.norm(legacy_adapter().compose(p)[:,:3],axis=-1)
    assert np.all(r>=np.exp(-1)-1e-7)
    b=normalization_boundary()
    assert b['zero_direction_execution']==[0]*5
    assert np.linalg.norm(b['subepsilon_direction_execution'][:3])<np.exp(-1)


def test_identical_preclip_directions_and_shared_scalar_coordinates():
    raw=np.random.default_rng(3).normal(size=(1000,6));r=probe(raw,.2)
    assert r['shared_preclip_direction_max_difference']<1e-12
    assert r['posthoc']['median']>.8 and r['native']['median']<.4
    assert r['normalization_floor_fraction']==0


def test_prior_speed_coordinate_mismatch_is_counted_not_called_unreachable():
    a=np.array([[.1,0,0,.2,-.3],[.8,0,0,.2,-.3],[0,0,0,.2,-.3]])
    r=replay_geometry(a)
    assert r['log_speed_below_actor_coordinate_bound_count']==2
    assert r['normalization_floor_count']==1
    assert 'escape branch' in r['scope']


def test_analytic_jacobian_matches_actual_mapper_and_finite_differences():
    r=jacobian_check()
    assert r['analytic_max_error']<2e-6 and r['finite_difference_max_error']<6e-5
    assert r['radial_direction_null_residual']<1e-6


@pytest.mark.parametrize('a',[np.empty((0,5)),np.zeros((2,4)),np.full((2,5),np.nan),np.full((2,5),1.1)])
def test_invalid_prior_data_rejected(a):
    with pytest.raises(ValueError):replay_geometry(a)


def test_no_global_bigger_actions_claim_at_standard_noise():
    raw=np.random.default_rng(4).normal(size=(10000,6));r=probe(raw,1.)
    assert r['posthoc']['mean']<r['native']['mean']
    assert r['actual_preclip_xyz_clipping_fraction']>.3
