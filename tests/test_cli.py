"""CLI import safety, explicit protocol, and filesystem behavior."""

import ast
import importlib
import json
import subprocess
import sys

import pytest

from src.cli import main
from src.config import WORKFLOWS, build_parser, parse_args
from src.utils import ROOT, create_run, read_selection, write_json


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_help_and_import(workflow, capsys):
    """Every supported entry point builds help and imports without running experiments."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser(workflow).parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "--protocol" in capsys.readouterr().out
    if workflow != "random_scores":
        importlib.import_module(f"scripts.{workflow}")


def test_protocol_and_old_boolean_aliases():
    """Protocol is mandatory and historical boolean spelling remains supported."""
    with pytest.raises(SystemExit):
        parse_args("train_rl", ["--env_id", "Swimmer-v4"])
    args = parse_args("train_rl", ["--protocol", "legacy", "--env_id", "Swimmer-v4", "--use_reservoir", "False"])
    assert args.units == 100 and not args.use_reservoir
    assert args.training_steps == 300000
    corrected = parse_args("train_all_selected", ["--protocol", "corrected"])
    assert corrected.units == 200 and corrected.policy_seeds == [0, 1, 2, 3, 4]
    assert corrected.del_obs and corrected.reset_res


@pytest.mark.parametrize(
    "arguments",
    [["--units", "0"], ["--res-lr", "2"], ["--rank-energy", "2"], ["--min-rank", "51"], ["--n-delays", "0"]],
)
def test_invalid_arguments(arguments):
    """Reject invalid counts and scientific parameters before execution."""
    workflow = "inputdsa_100_reservoirs_mackey_glass_ou"
    with pytest.raises(SystemExit):
        parse_args(workflow, ["--protocol", "legacy", *arguments])


def test_dry_run_from_other_cwd(tmp_path):
    """Repository defaults work outside the checkout; dry-run creates no artifacts."""
    destination = tmp_path / "absent"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inputdsa_100_reservoirs_H1.py"),
            "--protocol",
            "legacy",
            "--dry-run",
            "--output-dir",
            str(destination),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["selection_rows"] == 100
    assert config["arguments"]["selected_csv"] == str(ROOT / "selected.csv")
    assert not destination.exists()


def test_no_run_reuse_without_explicit_overwrite(tmp_path):
    """Existing outputs survive accidental reruns and receive a clear error."""
    args = parse_args("generate_fig_1", ["--protocol", "legacy", "--output-dir", str(tmp_path), "--run-id", "fixed"])
    run = create_run(args, "generate_fig_1")
    marker = run / "figures" / "keep.png"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        create_run(args, "generate_fig_1")
    assert marker.read_bytes() == b"original"


def test_corrected_rejects_historical_selection():
    """Historical selections cannot silently enter the corrected protocol."""
    with pytest.raises(ValueError, match="corrected selection"):
        main(
            "train_all_selected", ["--protocol", "corrected", "--selected-csv", str(ROOT / "selected.csv"), "--dry-run"]
        )


def test_original_selection_identity():
    """Literal null strata and original row identifiers survive stable sorting."""
    root = read_selection(ROOT / "selected.csv")
    alternative = read_selection(ROOT / "scripts/selection/selected.csv")
    assert len(root) == len(alternative) == 100
    assert (root.stratum == "null").sum() == 20
    assert len(set(root.seed) & set(alternative.seed)) == 24
    sorted_frame = root.sort_values("stratum")
    assert set(sorted_frame.csv_idx) == set(range(100))


@pytest.mark.parametrize("mismatch", ["protocol", "units", "matrix_hash"])
def test_corrected_pool_provenance(tmp_path, parameter_row, mismatch):
    """Reject historical or unrelated descriptor pools before analysis writes anything."""
    import pandas as pd

    from src.reservoirs import build_corrected, save_reservoir

    artifact = build_corrected(parameter_row, units=12)
    save_reservoir(tmp_path / "matrix.npz", artifact)
    selected = pd.DataFrame(
        [
            {
                **parameter_row,
                "protocol": "corrected",
                "units": 12,
                "matrix_hash": str(artifact["matrix_hash"]),
                "matrix_file": "matrix.npz",
            }
        ]
    )
    selected.to_csv(tmp_path / "selected.csv", index=False)
    pool = selected.copy()
    if mismatch == "protocol":
        pool = pool.drop(columns="protocol")
    elif mismatch == "units":
        pool["units"] = 200
    else:
        pool["matrix_hash"] = "another-reservoir"
    pool.to_csv(tmp_path / "descriptors.csv", index=False)
    with pytest.raises(ValueError, match="pool|neuron count"):
        main(
            "descriptor_space_plot_performance",
            [
                "--protocol",
                "corrected",
                "--selected-csv",
                str(tmp_path / "selected.csv"),
                "--dry-run",
                "--output-dir",
                str(tmp_path / "absent"),
            ],
        )
    assert not (tmp_path / "absent").exists()


def test_manifest_uses_strict_json(tmp_path):
    """Infinity used for equal sizes is metadata text, not invalid JSON syntax."""
    import numpy as np

    target = tmp_path / "manifest.json"
    write_json(target, {"alpha": [np.inf], "undefined": np.nan})
    value = json.loads(target.read_text())
    assert value == {"alpha": ["inf"], "undefined": None}


def test_first_party_docstrings():
    """Every first-party function and class has introductory documentation."""
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "er_mrl"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text())
            assert ast.get_docstring(tree), path
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    assert ast.get_docstring(node), f"{path}:{node.lineno} {node.name}"
