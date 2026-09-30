"""Train the first configuration per stratum with independent policy replicates.

Example (from repository root)::

    python scripts/train_and_plot.py --protocol legacy --seeds 5

Reads an explicit selection. Writes models, logs, and replicate mean/std learning curves; legacy seeds also change reservoir weights.
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
    return workflow_parser("train_and_plot")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("train_and_plot", argv)


if __name__ == "__main__":
    main()
