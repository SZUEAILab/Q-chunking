import numpy as np

from research.stage_task import (
    APPROACH_RADIUS,
    BASE_SIGMA,
    DS_TANGENTIAL_STD,
    direction_speed_actions,
    local_covariance_volume,
    matching_radial_sigma,
    within_budget_probability,
)


def test_direction_speed_allocation_matches_cartesian_local_volume():
    radial_sigma = matching_radial_sigma(BASE_SIGMA, DS_TANGENTIAL_STD)

    np.testing.assert_allclose(
        local_covariance_volume(DS_TANGENTIAL_STD, radial_sigma),
        BASE_SIGMA**3,
    )


def test_within_budget_probability_matches_geometric_complement():
    np.testing.assert_allclose(within_budget_probability(0.1, 2), 0.19)


def test_zero_noise_direction_speed_action_is_exact_center():
    noise = np.zeros((3, 3))
    actions = direction_speed_actions(
        np.asarray([1.0, 0.0, 0.0]),
        APPROACH_RADIUS,
        noise,
        direction_scale=0.0,
        radial_sigma=0.0,
    )

    np.testing.assert_allclose(actions, [[APPROACH_RADIUS, 0.0, 0.0]] * 3)
