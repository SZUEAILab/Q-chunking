"""One-step OGBench adaptation of the pinned upstream RLPD learner recipe."""

from absl import app

import main_online
from agents.rlpd_reference import RLPDReferenceAgent


main_online.agents["rlpd_reference"] = RLPDReferenceAgent
main_online.FLAGS.set_default("agent", "agents/rlpd_reference.py")
main_online.FLAGS.set_default("horizon_length", 1)
main_online.FLAGS.set_default("utd_ratio", 20)


def main(argv):
    flags = main_online.FLAGS
    if flags.agent.agent_name != "rlpd_reference":
        raise ValueError("Reference entry point requires --agent=agents/rlpd_reference.py.")
    if flags.horizon_length != 1 or flags.ds_mode != "none" or flags.stage_balanced_sampling:
        raise ValueError("Reference driver requires H1 native actions and uniform replay sampling.")
    if flags.discount != flags.agent.discount:
        raise ValueError("Driver and learner discount must agree.")
    main_online.main(argv)


if __name__ == "__main__":
    app.run(main)
