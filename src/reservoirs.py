"""Canonical corrected reservoir construction and immutable matrix artifacts.

The factory samples structure once. RL and synthetic experiments load its W,
assignments, and block matrices instead of resampling from a lossy CSV mapping.
"""

from pathlib import Path

import numpy as np

from src.utils import matrix_hash


def fixed_density(rho, p0, assignments, core=False, f=0.5):
    """Solve expected density over actual unordered node pairs, without clipping.

    Raises ValueError when the requested ratio and density are infeasible.
    """
    counts = np.bincount(assignments)
    count = len(counts)
    ratios = np.ones((count, count))
    np.fill_diagonal(ratios, rho)
    if core:
        ratios[:] = 1
        ratios[0, :] = ratios[:, 0] = 1 + f * (rho - 1)
        ratios[0, 0] = rho
    pairs = np.outer(counts, counts).astype(float)
    np.fill_diagonal(pairs, counts * (counts - 1))
    denominator = np.sum(pairs * ratios)
    if denominator <= 0:
        raise ValueError("Density constraints have no eligible edges")
    probabilities = ratios * p0 * pairs.sum() / denominator
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Infeasible density/topology ratio; resample rather than clip")
    return probabilities


def build_corrected(row, units=200, sigma_base=0.1, sr=0.9):
    """Sample a paper-aligned WSBM and return its complete realized artifact.

    centre_u and contrast_u are in sigma_base units; contrast_u is half the
    within-minus-between contrast. The null arm has uniform block parameters.
    """
    from reservoirpy.nodes.wsbm_esn import WSBMESN, dirichlet_sizes, equal_sizes

    seed, count = int(row["seed"]), int(row["K"])
    if units < count or units < 2 or sigma_base <= 0 or not 0 < sr < 1:
        raise ValueError("Require units >= K, sigma_base > 0, and 0 < sr < 1")
    rng = np.random.default_rng(seed)
    alpha = float(row["size_alpha"])
    if row["stratum"] == "core_periphery":
        core = max(1, int(round(units * float(row["core_fraction"]))))
        if units - core < count - 1:
            raise ValueError("Core allocation leaves an empty community")
        assignments = np.concatenate([np.zeros(core, dtype=int), equal_sizes(units - core, count - 1) + 1])
    else:
        assignments = equal_sizes(units, count) if not np.isfinite(alpha) else dirichlet_sizes(units, count, alpha, rng)
    hi = (float(row["centre_u"]) + float(row["contrast_u"])) * sigma_base
    lo = (float(row["centre_u"]) - float(row["contrast_u"])) * sigma_base
    dispersion = float(row["sigma_ratio"])
    sigma = np.full((count, count), sigma_base * count / (dispersion + count - 1))
    np.fill_diagonal(sigma, sigma[0, 0] * dispersion)
    connectivity = fixed_density(
        float(row["rho"]), float(row["p0"]), assignments, row["stratum"] == "core_periphery", float(row["f"])
    )
    motif = row["stratum"] if row["stratum"] in {"mixed", "core_periphery"} else "assortative"
    node = WSBMESN(
        units=units,
        assignments=assignments,
        n_communities=count,
        motif=motif,
        hi=hi,
        lo=lo,
        mid=lo + float(row["f"]) * (hi - lo),
        sigma=sigma,
        connectivity=connectivity,
        symmetric=True,
        p_negative=0,
        seed=seed,
        sr=sr,
    )
    node.initialize(np.zeros((1, 1)))
    matrix = np.asarray(node.W)
    if not np.isfinite(matrix).all() or np.max(np.abs(matrix)) == 0:
        raise ValueError("Degenerate recurrent matrix")
    return {
        "W": matrix,
        "assignments": assignments,
        "omega": node.omega,
        "sigma": sigma,
        "connectivity": connectivity,
        "units": np.array(units),
        "reservoir_seed": np.array(seed),
        "spectral_radius": np.array(sr),
        "matrix_hash": np.array(matrix_hash(matrix)),
        "protocol": np.array("corrected"),
    }


def save_reservoir(path, artifact):
    """Persist numeric arrays and Unicode metadata without Python pickles."""
    np.savez_compressed(path, **artifact)


def load_reservoir(path, row=None, units=None, sr=None):
    """Load and verify a corrected artifact against expected row and dimensions."""
    with np.load(Path(path), allow_pickle=False) as data:
        artifact = {key: data[key] for key in data.files}
    matrix = artifact["W"]
    if matrix_hash(matrix) != str(artifact["matrix_hash"]):
        raise ValueError(f"Matrix hash mismatch: {path}")
    count = int(artifact["units"])
    if matrix.shape != (count, count) or not np.allclose(matrix, matrix.T) or np.any(np.diag(matrix)):
        raise ValueError("Invalid recurrent matrix geometry")
    if units is not None and units != count:
        raise ValueError(f"Selected matrix has {count} units, requested {units}")
    if row is not None and int(row["seed"]) != int(artifact["reservoir_seed"]):
        raise ValueError("Reservoir artifact belongs to a different row")
    if row is not None and "matrix_hash" in row and str(row["matrix_hash"]) != str(artifact["matrix_hash"]):
        raise ValueError("Selected CSV and reservoir matrix hashes disagree")
    if sr is not None and not np.isclose(sr, float(artifact["spectral_radius"])):
        raise ValueError("Requested spectral radius differs from selected matrix")
    return artifact


def reservoir_path(selected_csv, row):
    """Resolve matrix paths relative to the selection CSV that declares them."""
    if "matrix_file" not in row or not row["matrix_file"]:
        raise ValueError("Corrected runs require a corrected selection with matrix_file metadata")
    return Path(selected_csv).parent / str(row["matrix_file"])
