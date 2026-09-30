"""Compare standardized structural distance with intrinsic state distance for all selected reservoirs.

Example (from repository root)::

    python scripts/inputdsa_100_reservoirs_H1.py --protocol legacy --tasks Swimmer-v4

Reads selection and trajectories. Writes five DSA matrices, diagnostics, pair tables, and scatter/hexbin figures. Historical H1 filename is not paper H1.
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
    return workflow_parser("inputdsa_100_reservoirs_H1")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("inputdsa_100_reservoirs_H1", argv)


if __name__ == "__main__":
    main()
