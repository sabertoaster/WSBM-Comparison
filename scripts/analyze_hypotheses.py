"""Test paper H1 dynamical convergence and H2 structural-distance/return association.

Example (from repository root)::

    python scripts/analyze_hypotheses.py --protocol corrected --selected-csv artifacts/corrected/selection/selection/selected.csv --evaluation-csv artifacts/corrected/collection/evaluation/episodes.csv --trajectories-csv artifacts/corrected/collection/trajectories/trajectories.csv

Reads selection, pool, evaluation episodes, and trajectories. Writes H1/H2 statistics with permutation p-values, pair tables, matrices, and figures.
All new outputs are isolated under artifacts/<protocol>/<run-id>/.
Use --help for parameters; --dry-run validates without writing or running.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser


def build_parser():
    """Return the documented argparse interface for this workflow."""
    return workflow_parser("analyze_hypotheses")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("analyze_hypotheses", argv)


if __name__ == "__main__":
    main()
