"""Shared artifact, tabular, logging, and numerical utilities.

Scripts use repository-relative defaults and caller-relative explicit paths.
No utility deletes files or starts experiments on import.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("Hopper-v4", "Walker2d-v4", "Swimmer-v4", "Ant-v4", "HalfCheetah-v4")
STRATA = ("assortative", "disassortative", "core_periphery", "mixed", "null")
DESCRIPTORS = (
    "modularity",
    "participation",
    "clustering",
    "efficiency",
    "cv_strength",
    "cv_betweenness",
    "lambda_ratio",
    "frac_negative",
)


def resolve_path(value, default):
    """Return an absolute path; explicit values are relative to the caller."""
    return Path(value).expanduser().resolve() if value is not None else ROOT / default


def file_hash(path):
    """Return the SHA256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_hash(matrix):
    """Hash shape, dtype, and contiguous array bytes for matrix identity checks."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.ascontiguousarray(matrix)
    if array.dtype.hasobject:
        raise ValueError("Matrix checksums require numeric arrays")
    digest = hashlib.sha256(f"{array.shape}:{array.dtype}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, data):
    """Write JSON metadata, converting paths and NumPy scalar values."""

    def convert(value):
        """Serialize scientific metadata without allowing executable objects."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None if np.isnan(value) else "inf" if value > 0 else "-inf"
        return value

    Path(path).write_text(json.dumps(convert(data), indent=2, allow_nan=False) + "\n")


def revision(path=ROOT):
    """Read a Git commit without changing the index or any Git objects."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def create_run(args, workflow):
    """Create an isolated run directory and manifest; reject accidental reuse."""
    root = resolve_path(args.output_dir, "artifacts")
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{workflow}-{uuid4().hex[:8]}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    path = root / args.protocol / run_id
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Run exists: {path}; use a new run-id or --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    for folder in ("selection", "models", "logs", "evaluation", "trajectories", "analysis", "figures"):
        (path / folder).mkdir(exist_ok=True)
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "scikit-learn",
        "networkx",
        "igraph",
        "reservoirpy",
        "gymnasium",
        "mujoco",
        "stable-baselines3",
        "dsa-metric",
        "tensorboard",
        "tensorflow",
        "matplotlib",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    selection = getattr(args, "selected_csv", None)
    source_files = [ROOT / "main.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for directory in ("src", "scripts", "er_mrl"):
        source_files.extend(sorted((ROOT / directory).rglob("*.py")))
    inputs = {}
    for key in ("selected_csv", "pool_csv", "models_csv", "evaluation_csv", "trajectories_csv"):
        value = getattr(args, key, None)
        if value and Path(value).is_file():
            inputs[key] = {"path": str(value), "sha256": file_hash(value)}
    write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "workflow": workflow,
            "arguments": vars(args),
            "run_dir": path,
            "revision": revision(),
            "dsa_revision": revision(ROOT / "DSA"),
            "versions": versions,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "source_sha256": {str(file.relative_to(ROOT)): file_hash(file) for file in source_files},
            "input_files": inputs,
            "selection_sha256": file_hash(selection) if selection and Path(selection).is_file() else None,
            "status": "started",
        },
    )
    print(f"Run directory: {path}", flush=True)
    return path


def finish_run(path, **metadata):
    """Merge result metadata into a completed run's manifest."""
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(status="completed", **metadata)
    write_json(path / "manifest.json", manifest)


def read_selection(path):
    """Load an explicit selection CSV and preserve original row identifiers.

    The literal stratum ``null`` must not become a missing value. Returns a
    DataFrame with unique integer ``csv_idx`` values, never inferred after sort.
    """
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"stratum", "K", "p0", "centre_u", "contrast_u", "sigma_ratio", "seed"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Selection missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Selection is empty")
    if "csv_idx" not in frame:
        frame.insert(0, "csv_idx", np.arange(len(frame)))
    for key in ("csv_idx", "K", "seed"):
        values = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{key} must contain finite integers")
        frame[key] = values.astype(int)
    if frame.csv_idx.duplicated().any() or (frame.csv_idx < 0).any():
        raise ValueError("csv_idx must be unique and nonnegative")
    if not frame.stratum.isin(STRATA).all():
        raise ValueError("Unknown or missing reservoir stratum")
    return frame


def legacy_config(row):
    """Reproduce historical CSV-to-RL parameter conversion exactly."""
    return {
        "csv_idx": int(row["csv_idx"]),
        "motif": row["stratum"],
        "n_communities": int(row["K"]),
        "connectivity": float(row["p0"]),
        "hi": float(row["centre_u"] + row["contrast_u"] / 2),
        "lo": float(row["centre_u"] - row["contrast_u"] / 2),
        "mid": float(row["centre_u"]),
        "sigma": float(row["sigma_ratio"] * row["centre_u"]),
        "p_negative": float(row["frac_negative"]),
        "seed": int(row["seed"]),
        "core_fraction": float(row.get("core_fraction", 0.2)),
    }


def get_random_seed():
    """Return the SLURM array seed, or zero outside an array job."""
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def experiment_name(task, motif, steps, csv_idx=None):
    """Return the historical experiment name and model prefix."""
    prefix = "PPO" if motif is None else f"RES_{motif}" + (f"_idx{csv_idx}" if csv_idx is not None else "") + "_PPO"
    return f"{task}_{prefix}_{steps}steps", prefix


def delay_embed(episodes, n_delays):
    """Stack per-episode delay embeddings without crossing episode boundaries."""
    if n_delays < 1:
        raise ValueError("n_delays must be positive")
    if isinstance(episodes, np.ndarray) and episodes.ndim == 2:
        episodes = [episodes]
    blocks = []
    for episode in episodes:
        array = np.asarray(episode, dtype=float)
        if array.ndim != 2 or not np.isfinite(array).all() or len(array) <= 2 * n_delays:
            raise ValueError("Each trajectory must be finite, 2D, and longer than twice n_delays")
        blocks.append(np.hstack([array[i : len(array) - n_delays + i + 1] for i in range(n_delays)]))
    if not blocks:
        raise ValueError("No episodes supplied")
    return np.vstack(blocks)


def choose_rank(Ys, n_delays, energy=0.99, min_rank=1, max_rank=50):
    """Return common capped SVD rank and uncapped per-system energy ranks.

    Systems may contain a single (T, features) array or a list of episodes.
    Reject degenerate inputs rather than silently fitting meaningless dynamics.
    """
    if not 0 < energy <= 1 or not 1 <= min_rank <= max_rank or not Ys:
        raise ValueError("Invalid energy, rank bounds, or empty systems")
    ranks, available = [], []
    for system in Ys:
        hankel = delay_embed(system, n_delays)
        singular = np.linalg.svd(hankel - hankel.mean(axis=0), compute_uv=False)
        total = np.sum(singular**2)
        if total <= np.finfo(float).eps:
            raise ValueError("Constant trajectory cannot support a dynamical rank")
        ranks.append(min(len(singular), int(np.searchsorted(np.cumsum(singular**2) / total, energy)) + 1))
        available.append(np.linalg.matrix_rank(hankel - hankel.mean(axis=0)))
    upper = min(max_rank, min(available))
    if upper < min_rank:
        raise ValueError("Trajectories do not support min_rank")
    return int(np.clip(max(ranks), min_rank, upper)), ranks


def standardize(values, reference=None):
    """Z-score finite descriptors against an explicit reference population."""
    values = np.asarray(values, dtype=float)
    reference = values if reference is None else np.asarray(reference, dtype=float)
    if values.ndim != 2 or reference.ndim != 2 or not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Descriptors must be finite 2D arrays")
    scale = reference.std(axis=0)
    return (values - reference.mean(axis=0)) / np.where(scale > 0, scale, 1)


def read_scalars(logdir, tag="rollout/ep_rew_mean"):
    """Read all events in one TensorBoard run; deduplicate by actual step.

    Sorted file order resolves repeated steps to the last recorded value.
    Files belonging to different policy seeds must be passed separately.
    """
    configure_tensorboard()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = {}
    for path in sorted(Path(logdir).rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                values[event.step] = event.value
    steps = np.array(sorted(values), dtype=int)
    return steps, np.array([values[step] for step in steps])


def configure_tensorboard():
    """Use TensorBoard's bundled TF stub for PyTorch scalar logging.

    This avoids loading a second CUDA runtime merely to read/write PPO scalars.
    It does not disable or replace the installed TensorFlow package.
    """
    import tensorboard.compat
    from tensorboard.compat import tensorflow_stub

    tensorboard.compat.tf = tensorflow_stub


def align_curves(curves):
    """Interpolate curves on recorded steps within their shared time interval."""
    curves = [(np.asarray(s), np.asarray(v)) for s, v in curves if len(s)]
    if not curves:
        return np.array([]), np.array([]), np.array([])
    lower, upper = max(s[0] for s, _ in curves), min(s[-1] for s, _ in curves)
    grid = np.unique(np.concatenate([s[(s >= lower) & (s <= upper)] for s, _ in curves]))
    if not len(grid):
        return grid, np.array([]), np.array([])
    values = np.array([np.interp(grid, s, v) for s, v in curves])
    return grid, values.mean(axis=0), values.std(axis=0)


def weighted_performance(steps, values, window_steps=10000, weight_type="linear"):
    """Compute the historical last-window reward score on sorted scalars."""
    if window_steps <= 0 or weight_type not in {"linear", "exponential", "mean"}:
        raise ValueError("Invalid ranking window or weight type")
    if not len(steps):
        return float("-inf")
    keep = steps >= steps[-1] - window_steps
    steps, values = steps[keep], values[keep]
    if weight_type == "linear":
        weights = np.ones(1) if len(steps) == 1 else np.clip((steps - steps[-1] + window_steps) / window_steps, 1e-6, 1)
    elif weight_type == "exponential":
        weights = np.exp((steps - steps[-1]) / (window_steps / 3))
    else:
        weights = np.ones(len(steps))
    return float(np.average(values, weights=weights))
