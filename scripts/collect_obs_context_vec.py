"""Evaluate checkpoints and collect reservoir input/context trajectories per episode.

Example (from repository root)::

    python scripts/collect_obs_context_vec.py --protocol legacy --tasks Swimmer-v4 --limit 2

Reads historical checkpoints or --models-csv. Writes episode returns and NPZ trajectories with an episode registry.
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
    return workflow_parser("collect_obs_context_vec")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("collect_obs_context_vec", argv)


if __name__ == "__main__":
    main()
