"""Plot post-update reservoir activity using PCA and observation/action/reward CEBRA.

Run ``uv run python scripts/plot_context_embeddings.py --help`` for options.
Legacy checkpoints retain their trained reservoir protocol. All outputs are
isolated under artifacts/<protocol>/<run-id>; --dry-run writes nothing.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser


def build_parser():
    """Return the shared workflow CLI."""
    return workflow_parser("plot_context_embeddings")


def main(argv=None):
    """Run extraction, ranking, and embeddings without training RL policies."""
    return run_workflow("plot_context_embeddings", argv)


if __name__ == "__main__":
    main()
