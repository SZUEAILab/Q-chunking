import os
from pathlib import Path
import subprocess
import sys

import pytest

from research.config_argv import config_parent_first


@pytest.mark.parametrize("parent", [["--agent=config.py"], ["--agent", "config.py"]])
def test_preserves_values_and_is_idempotent(parent):
    argv = ["entry.py", "--agent.width=7", "--seed=10", *parent, "--agent.flag=True"]
    result = config_parent_first(argv)
    assert result == ["entry.py", *parent, "--agent.width=7", "--seed=10", "--agent.flag=True"]
    assert config_parent_first(result) == result


@pytest.mark.parametrize("tail", [["--agent"], ["--agent="], ["--agent", "--seed=1"], ["--agent=a", "--agent=b"]])
def test_rejects_ambiguous_parent(tail):
    with pytest.raises(ValueError):
        config_parent_first(["entry.py", *tail])


def test_real_ml_collections_parser(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("from ml_collections import ConfigDict\ndef get_config():\n return ConfigDict(dict(width=2, flag=False))\n")
    driver = tmp_path / "driver.py"
    driver.write_text("""import sys
from research.config_argv import config_parent_first
if sys.argv[1] == 'repair': sys.argv[:] = config_parent_first([sys.argv[0], *sys.argv[2:]])
else: del sys.argv[1]
from absl import app
from ml_collections import config_flags
config_flags.DEFINE_config_file('agent')
def main(argv):
 from absl import flags
 assert flags.FLAGS.agent.width == 7 and flags.FLAGS.agent.flag is True
 print('parsed_expected_values')
app.run(main)
""")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    tail = ["--agent.width=7", "--agent.flag=True", f"--agent={config}"]
    failed = subprocess.run([sys.executable, str(driver), "original", *tail], env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and "before a value for --agent" in failed.stderr
    repaired = subprocess.run([sys.executable, str(driver), "repair", *tail], env=env, capture_output=True, text=True)
    assert repaired.returncode == 0, repaired.stderr
    assert "parsed_expected_values" in repaired.stdout
