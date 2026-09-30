"""Compare structural/state distances for top and bottom reservoir groups.

Example (from repository root)::

    python scripts/inputdsa_top_bottom_10_reservoirs_H1.py --protocol legacy --tasks Walker2d-v4

Reads selection, performance source, and trajectories. Writes ordered DSA matrices, pair tables, and figures.
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
    return workflow_parser("inputdsa_top_bottom_10_reservoirs_H1")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("inputdsa_top_bottom_10_reservoirs_H1", argv)


if __name__ == "__main__":
    main()
