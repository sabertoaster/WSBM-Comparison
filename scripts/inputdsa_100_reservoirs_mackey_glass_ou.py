"""Compare all selected reservoirs under shared synthetic Mackey-Glass and OU input.

Example (from repository root)::

    python scripts/inputdsa_100_reservoirs_mackey_glass_ou.py --protocol legacy --conditions ou_noise --limit 5

Reads selection; generates input. Writes ordered DSA matrices, diagnostics, and figures. Legacy uses the historical subspace backend.
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
    return workflow_parser("inputdsa_100_reservoirs_mackey_glass_ou")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("inputdsa_100_reservoirs_mackey_glass_ou", argv)


if __name__ == "__main__":
    main()
