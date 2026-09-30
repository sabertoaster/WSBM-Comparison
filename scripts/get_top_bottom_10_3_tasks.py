"""Rank reservoirs by deterministic evaluation returns or legacy reward windows.

Example (from repository root)::

    python scripts/get_top_bottom_10_3_tasks.py --protocol legacy --tasks Swimmer-v4

Reads selection and evaluation CSV or TensorBoard logs. Writes performance and top/bottom ranking tables.
All new outputs are isolated under artifacts/<protocol>/<run-id>/.
Use --help for parameters; --dry-run validates without writing or running.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.analysis import get_performance as get_performance
from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser
from src.utils import weighted_performance


def compute_weighted_mean(scalars, window_steps=10000, weight_type="linear"):
    """Compatibility adapter for TensorBoard scalar objects."""
    ordered = sorted(scalars, key=lambda event: event.step)
    return weighted_performance(
        np.array([event.step for event in ordered]),
        np.array([event.value for event in ordered]),
        window_steps,
        weight_type,
    )


def build_parser():
    """Return the documented argparse interface for this workflow."""
    return workflow_parser("get_top_bottom_10_3_tasks")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("get_top_bottom_10_3_tasks", argv)


if __name__ == "__main__":
    main()
