"""Compare five InputDSA metrics across the selected reservoirs, grouped by stratum.

Example (from repository root)::

    python scripts/inputdsa_100_reservoirs_H2.py --protocol legacy --tasks Swimmer-v4

Reads selection and trajectories. Writes ordered distance matrices, diagnostics, and figures. This historical H2 filename does not test paper H2.
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
    return workflow_parser("inputdsa_100_reservoirs_H2")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("inputdsa_100_reservoirs_H2", argv)


if __name__ == "__main__":
    main()
