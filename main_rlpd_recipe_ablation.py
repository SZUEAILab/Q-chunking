"""Fresh-cohort native recipe intervention entry; no changes to frozen drivers."""

from absl import app

import main_online
from agents.rlpd_recipe_ablation import RLPDRecipeAblationAgent

main_online.agents['rlpd_recipe_ablation'] = RLPDRecipeAblationAgent
main_online.FLAGS.set_default('agent', 'agents/rlpd_recipe_ablation.py')
main_online.FLAGS.set_default('horizon_length', 1)


def validate_flags(flags):
    if (flags.agent.agent_name != 'rlpd_recipe_ablation' or flags.horizon_length != 1
            or flags.ds_mode != 'none' or flags.stage_balanced_sampling
            or flags.discount != flags.agent.discount):
        raise ValueError('Recipe ablation requires the native H1 matched-data driver.')
    if flags.agent.recipe_target_reduction not in ('min', 'mean'):
        raise ValueError('Unknown target reduction.')
    if flags.agent.num_qs != 10 or flags.agent.num_min_qs != 2:
        raise ValueError('Keep the ten-Q ensemble and two-head target draw fixed.')


def main(argv):
    validate_flags(main_online.FLAGS)
    main_online.main(argv)


if __name__ == '__main__':
    app.run(main)
