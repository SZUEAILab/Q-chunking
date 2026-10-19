"""H1 original posthoc with the original RLPD learner recipe and driver bridge."""

from absl import app

import main_online
from agents.rlpd_posthoc_reference import RLPDPosthocReferenceAgent


main_online.agents["rlpd_posthoc_reference"] = RLPDPosthocReferenceAgent
main_online.FLAGS.set_default("agent", "agents/rlpd_posthoc_reference.py")
main_online.FLAGS.set_default("horizon_length", 1)
main_online.FLAGS.set_default("utd_ratio", 20)
main_online.FLAGS.set_default("ds_mode", "adapter")
main_online.FLAGS.set_default("ds_adapter_speed_squash", "tanh")
main_online.FLAGS.set_default("ds_adapter_speed_interpretation", "log")


def validate_driver(flags):
    if flags.agent.agent_name != "rlpd_posthoc_reference":
        raise ValueError("Posthoc reference entry requires --agent=agents/rlpd_posthoc_reference.py.")
    if flags.horizon_length != 1 or flags.ds_mode != "adapter" or flags.stage_balanced_sampling:
        raise ValueError("Posthoc reference requires H1 adapter actions and uniform replay sampling.")
    bound = flags.agent.ds_speed_bound if flags.ds_speed_bound == "config" else flags.ds_speed_bound
    if (flags.ds_adapter_representation != "cartesian" or flags.ds_adapter_speed_squash != "tanh"
            or flags.ds_adapter_speed_interpretation != "log" or bound != "sphere"
            or flags.ds_adapter_critic_space != "latent"):
        raise ValueError("Posthoc reference driver requires original Cartesian tanh/log/sphere latent recipe.")
    if flags.discount != flags.agent.discount:
        raise ValueError("Driver and learner discount must agree.")


def main(argv):
    validate_driver(main_online.FLAGS)
    main_online.main(argv)


if __name__ == "__main__":
    app.run(main)
