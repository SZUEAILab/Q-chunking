import jax
import numpy as np

from research.shell_offpolicy import policy_log_prob, sample_all
from research.shell import (
    CONDITIONS,
    PROJECTED_TANGENT_CONDITION,
    initial_parameters,
    parameter_count,
    sample_policy,
)


def test_projected_tangent_stage_matches_ds_stage_capacity():
    ds = next(condition for condition in CONDITIONS if condition.name == "DS stage")
    assert parameter_count(PROJECTED_TANGENT_CONDITION) == parameter_count(ds) == 8


def test_projected_tangent_replay_density_matches_sampler_density():
    params = initial_parameters(PROJECTED_TANGENT_CONDITION, 1)
    key = jax.random.PRNGKey(12)
    actions = sample_all(params, key[None, :], PROJECTED_TANGENT_CONDITION, 2048)
    replay_log_prob = policy_log_prob(
        params, actions, PROJECTED_TANGENT_CONDITION
    )
    _, sampler_log_prob = sample_policy(
        jax.tree_util.tree_map(lambda value: value[0], params),
        key,
        PROJECTED_TANGENT_CONDITION,
        2048,
    )
    np.testing.assert_allclose(
        np.asarray(replay_log_prob[0]), np.asarray(sampler_log_prob), atol=2e-5
    )
    assert np.isfinite(np.asarray(replay_log_prob)).all()


def test_projected_tangent_sampling_preserves_explicit_radial_scale():
    params = initial_parameters(PROJECTED_TANGENT_CONDITION, 1)
    actions = sample_all(
        params,
        jax.random.PRNGKey(31)[None, :],
        PROJECTED_TANGENT_CONDITION,
        8192,
    )
    radii = np.linalg.norm(np.asarray(actions[0]), axis=-1)
    expected = np.exp(np.asarray(params["log_scale"])[0, :, 1])
    np.testing.assert_allclose(radii.std(axis=1), expected, rtol=0.06)
