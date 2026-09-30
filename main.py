"""Unified research CLI.

Example: python main.py select_reservoirs --protocol corrected --dry-run
Use python main.py --help to list workflows. Existing scripts remain supported.
"""

from src.cli import entrypoint


def main(argv=None):
    """Run a named research workflow."""
    return entrypoint(argv)


if __name__ == "__main__":
    main()
