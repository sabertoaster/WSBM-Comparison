"""Draw the four canonical reservoir motifs using a shared connection-strength color scale.

Example (from repository root)::

    python scripts/generate_fig_1.py --protocol legacy --run-id motifs

No data input. Writes figures/wsbm_motifs.png; the existing myplot.png is preserved.
All new outputs are isolated under artifacts/<protocol>/<run-id>/.
Use --help for parameters; --dry-run validates without writing or running.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main as run_workflow
from src.config import build_parser as workflow_parser


def __getattr__(name):
    """Preserve the historical visualize_esn plotting helper on demand."""
    if name == "visualize_esn":
        from src.figures import visualize_esn

        return visualize_esn
    raise AttributeError(name)


def build_parser():
    """Return the documented argparse interface for this workflow."""
    return workflow_parser("generate_fig_1")


def main(argv=None):
    """Run the workflow with optional explicit command-line arguments."""
    return run_workflow("generate_fig_1", argv)


if __name__ == "__main__":
    main()
