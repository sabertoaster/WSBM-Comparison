"""Evaluate fixed trained policies on common seeded deterministic episodes.

Example (from repository root)::

    python scripts/evaluate.py --protocol corrected --selected-csv artifacts/corrected/selection/selection/selected.csv --models-csv artifacts/corrected/training/models/models.csv

Reads selection and a model registry. Writes per-episode returns and per-policy summaries.
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
    return workflow_parser("evaluate")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("evaluate", argv)


if __name__ == "__main__":
    main()
