"""CLI execution boundary: validate first, then create an isolated experiment."""

import json
from pathlib import Path

from src.config import WORKFLOWS, parse_args
from src.utils import create_run, finish_run, read_selection, write_json


def main(workflow, argv=None):
    """Run a workflow; dry-run writes nothing and failures remain in its manifest."""
    args = parse_args(workflow, argv)
    needs_selection = workflow not in {"select_reservoirs", "generate_fig_1", "random_scores"}
    if (
        workflow == "train_rl"
        and args.protocol == "legacy"
        and args.selected_csv.name == "selected.csv"
        and getattr(args, "csv_idx", None) is None
    ):
        needs_selection = False
    frame = None
    if needs_selection:
        if args.selected_csv is None:
            raise ValueError("Corrected workflows require --selected-csv from a corrected selection run")
        frame = read_selection(args.selected_csv)
        if args.protocol == "corrected":
            if "protocol" not in frame or not frame.protocol.eq("corrected").all():
                raise ValueError("Corrected runs require a corrected selection; historical CSVs are legacy data")
            if args.requested_units is None and "units" in frame:
                sizes = frame.units.unique()
                if len(sizes) != 1:
                    raise ValueError("Selection mixes reservoir sizes")
                args.units = int(sizes[0])
            if workflow == "train_rl" and getattr(args, "csv_idx", None) is None:
                raise ValueError("Corrected train_rl requires --csv-idx for one selected reservoir")
        if getattr(args, "csv_idx", None) is not None:
            frame = frame[frame.csv_idx == args.csv_idx]
            if frame.empty:
                raise ValueError("csv-idx does not exist in this selection")
        if args.limit is not None:
            frame = frame.head(args.limit)
        if args.protocol == "corrected":
            from src.reservoirs import load_reservoir, reservoir_path

            for _, row in frame.iterrows():
                load_reservoir(reservoir_path(args.selected_csv, row), row, args.units)
            if args.pool_csv is None and workflow in {
                "analyze_hypotheses",
                "descriptor_space_plot_performance",
                "inputdsa_100_reservoirs_H1",
                "inputdsa_top_bottom_10_reservoirs_H1",
            }:
                args.pool_csv = Path(args.selected_csv).parent / "descriptors.csv"
            if args.pool_csv is not None:
                pool = read_selection(args.pool_csv)
                if "protocol" not in pool or not pool.protocol.eq("corrected").all():
                    raise ValueError("Corrected analysis requires a corrected descriptor pool")
                if "units" not in pool or not pool.units.eq(args.units).all():
                    raise ValueError("Descriptor pool and selection must use the same neuron count")
                if "matrix_hash" not in pool or not frame.matrix_hash.isin(pool.matrix_hash).all():
                    raise ValueError("Descriptor pool does not contain the selected reservoir matrices")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "workflow": workflow,
                    "arguments": vars(args),
                    "selection_rows": len(frame) if frame is not None else None,
                },
                default=str,
                indent=2,
            )
        )
        return None
    output = create_run(args, workflow)
    try:
        from src.workflows import dispatch

        metadata = dispatch(args, frame, output, workflow)
        finish_run(output, **metadata)
    except Exception as error:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise
    return output


def entrypoint(argv=None):
    """Select a documented workflow through the repository's unified main.py."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="WSBM/InputDSA research workflows")
    parser.add_argument("workflow", choices=tuple(WORKFLOWS))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"]:
        parser.print_help()
        return None
    selected = parser.parse_args(arguments[:1])
    return main(selected.workflow, arguments[1:])
