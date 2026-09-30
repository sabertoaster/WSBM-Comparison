"""Train one PPO configuration with an optional reservoir encoder.

Example (from repository root)::

    python scripts/train_rl.py --protocol legacy --env-id Swimmer-v4 --training-steps 1000

Corrected mode requires --selected-csv and --csv-idx. Writes models, config, logs, registry, and learning curves.
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
    return workflow_parser("train_rl")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("train_rl", argv)


if __name__ == "__main__":
    main()
