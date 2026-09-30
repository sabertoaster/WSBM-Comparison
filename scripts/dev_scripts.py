"""Compare top/bottom reservoir dynamics and test paper H1 using group-label permutations.

Example (from repository root)::

    python scripts/dev_scripts.py --protocol legacy Swimmer-v4 --permutations 9999

Reads selection, rankings, and trajectories. Writes five DSA matrices, H1 test tables, diagnostics, and figures.
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
    return workflow_parser("dev_scripts")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("dev_scripts", argv)


if __name__ == "__main__":
    main()
