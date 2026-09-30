"""Compare one reservoir per stratum driven by Mackey-Glass or OU inputs.

Example (from repository root)::

    python scripts/inputdsa_5_systems_mackey_glass_ou.py --protocol legacy --conditions ou_noise

Reads selection; generates synthetic input. Writes five DSA matrices per input condition, diagnostics, and figures.
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
    return workflow_parser("inputdsa_5_systems_mackey_glass_ou")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("inputdsa_5_systems_mackey_glass_ou", argv)


if __name__ == "__main__":
    main()
