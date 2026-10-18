"""Locations for generated experiment artifacts, outside the source checkout."""

import os
from pathlib import Path


def data_root() -> Path:
    """Use QC_DATA_ROOT to select another mounted disk on a remote host."""
    return Path(os.environ.get("QC_DATA_ROOT", "/data/chainpray/qc")).expanduser().resolve()


def results_dir() -> Path:
    return data_root() / "results"


def reports_dir() -> Path:
    return data_root() / "reports" / "analysis"
