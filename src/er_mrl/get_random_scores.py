"""Evaluate seeded random-action agents without running anything on import.

Example: python -m src.er_mrl.get_random_scores --protocol legacy --tasks Swimmer-v4 --evaluation-episodes 10
Writes per-episode returns and legacy mean-step rewards inside an isolated run.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser


def build_parser():
    """Build the random-policy evaluation parser."""
    return workflow_parser("random_scores")


def main(argv=None):
    """Evaluate random policies using the requested task and seed settings."""
    return run_workflow("random_scores", argv)


if __name__ == "__main__":
    main()
