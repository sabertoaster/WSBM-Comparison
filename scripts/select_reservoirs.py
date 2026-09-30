"""Sample WSBM structures, calculate eight graph descriptors, and MaxMin-select a stratified subset.

Example (from repository root)::

    python scripts/select_reservoirs.py --protocol corrected --run-id selection

No input data. Writes selection CSVs, corrected matrices, PCA arrays, and a figure.
All new outputs are isolated under artifacts/<protocol>/<run-id>/.
Use --help for parameters; --dry-run validates without writing or running.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser


def __getattr__(name):
    """Lazily preserve historical structural helper imports."""
    if name in {"DESCRIPTORS", "STRATA"}:
        from src import utils

        return getattr(utils, name)
    if name in {"build", "density_matrix", "descriptors", "maxmin", "sample_params"}:
        from src import selection

        return getattr(selection, name)
    raise AttributeError(name)


def build_parser():
    """Return the documented argparse interface for this workflow."""
    return workflow_parser("select_reservoirs")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("select_reservoirs", argv)


if __name__ == "__main__":
    main()
