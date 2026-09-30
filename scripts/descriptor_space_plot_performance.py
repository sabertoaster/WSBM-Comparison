"""Project reservoirs onto full-pool structural PCA and color by performance.

Example (from repository root)::

    python scripts/descriptor_space_plot_performance.py --protocol legacy Swimmer-v4

Reads selection, full descriptors pool, and performance source. Writes PCA scatter figures and performance tables.
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
    return workflow_parser("descriptor_space_plot_performance")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("descriptor_space_plot_performance", argv)


if __name__ == "__main__":
    main()
