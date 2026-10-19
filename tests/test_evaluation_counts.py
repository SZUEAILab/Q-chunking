"""Observed counts must follow completed non-video episodes and terminal labels."""
import copy

import numpy as np
import pytest

from evaluation import evaluate, observed_evaluation_counts
from test_evaluation_seeding import _Agent, _OneStepEnv


class RenderEnv(_OneStepEnv):
    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


def test_real_evaluator_excludes_video_and_matches_aggregate():
    env = RenderEnv()
    stats, trajectories, videos = evaluate(_Agent(), env, num_eval_episodes=3, num_video_episodes=2,
                                          action_dim=1, video_frame_skip=1, eval_seed=123)
    counts = observed_evaluation_counts(stats, trajectories, expected_episodes=3)
    assert len(env.actions) == 5 and len(videos) == 2
    assert counts['evaluation_episodes_observed'] == 3
    assert counts['evaluation_successes_observed'] == sum(int(a[0] > 0) for a in env.actions[:3])
    assert stats['success'] == counts['evaluation_successes_observed']/3


def test_success_uses_terminal_info_not_ever_success():
    trajectories = [dict(done=[False, True], info=[dict(success=1.), dict(success=0.)])]
    assert observed_evaluation_counts({'success':0.}, trajectories, expected_episodes=1) == dict(
        evaluation_episodes_observed=1, evaluation_successes_observed=0)


def test_float32_success_mean_rounding_is_accepted():
    labels = np.array([1., 1., 1., 0., 0.], dtype=np.float32)
    trajectories = [dict(done=[True],info=[dict(success=x)]) for x in labels]
    counts = observed_evaluation_counts({'success':labels.mean()}, trajectories, expected_episodes=5)
    assert counts == dict(evaluation_episodes_observed=5, evaluation_successes_observed=3)


@pytest.mark.parametrize('kind', ['count', 'unfinished', 'multiple_episodes', 'missing_info', 'missing_label',
                                'nan_label', 'soft_label', 'array_label', 'wrong_aggregate', 'nan_aggregate'])
def test_incomplete_or_inconsistent_evaluation_is_rejected(kind):
    trajectories = [dict(done=[True], info=[dict(success=1.)]),dict(done=[True], info=[dict(success=0.)])]
    stats, expected = {'success':.5}, 2
    if kind == 'count': expected=3
    elif kind == 'unfinished': trajectories[0]['done']=[False]
    elif kind == 'multiple_episodes': trajectories[0]=dict(done=[True,True],info=[dict(success=1.)]*2)
    elif kind == 'missing_info': trajectories[0]['info']=[]
    elif kind == 'missing_label': trajectories[0]['info']=[{}]
    elif kind == 'nan_label': trajectories[0]['info']=[dict(success=float('nan'))]
    elif kind == 'soft_label': trajectories[0]['info']=[dict(success=.3)]
    elif kind == 'array_label': trajectories[0]['info']=[dict(success=[1.])]
    elif kind == 'wrong_aggregate': stats['success']=1.
    elif kind == 'nan_aggregate': stats['success']=float('nan')
    with pytest.raises(ValueError): observed_evaluation_counts(stats, trajectories, expected_episodes=expected)


def test_environments_without_success_do_not_get_fabricated_failure_counts():
    stats = {'return':12.}
    trajectories = [dict(done=[True],info=[dict(episode={'return':12.})])]
    before = copy.deepcopy((stats,trajectories))
    assert observed_evaluation_counts(stats, trajectories, expected_episodes=1) == {'evaluation_episodes_observed':1}
    assert (stats,trajectories) == before


def test_zero_evaluation_episodes_has_no_success_rate():
    assert observed_evaluation_counts({}, [], expected_episodes=0) == {'evaluation_episodes_observed':0}
