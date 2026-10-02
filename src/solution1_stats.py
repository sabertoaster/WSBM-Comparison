"""Dependent-pair inference and matched training-seed uncertainty for Solution 1."""

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf

from src.utils import standardize

PRIMARY = ("state_joint", "control_joint")


def correlations(x, y):
    """Compute row-wise Spearman coefficients, retaining undefined rows as NaN."""
    x, y = np.broadcast_arrays(np.atleast_2d(x), np.atleast_2d(y))
    a, b = rankdata(x, axis=-1), rankdata(y, axis=-1)
    a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide(np.sum(a * b, axis=-1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)


def label_permutations(n, count, seed, families=None):
    """Generate whole-reservoir label exchanges, optionally within families."""
    rng = np.random.default_rng(seed)
    result = np.tile(np.arange(n), (count, 1))
    blocks = [np.arange(n)] if families is None else [np.flatnonzero(families == f) for f in np.unique(families)]
    for row in result:
        for block in blocks:
            row[block] = rng.permutation(block)
    return result


def permutation_association(returns, matrix, permutations, structural=None):
    """Test H1 by performance-label exchange, or Fig. 3B by joint row/column exchange."""
    upper = np.triu_indices(len(matrix), 1)
    if structural is None:
        predictor = np.minimum(returns[upper[0]], returns[upper[1]])
        shuffled = returns[permutations]
        null = correlations(np.minimum(shuffled[:, upper[0]], shuffled[:, upper[1]]), matrix[upper])
    else:
        predictor = structural[upper]
        null = correlations(predictor, matrix[permutations[:, upper[0]], permutations[:, upper[1]]])
    observed = float(correlations(predictor, matrix[upper])[0])
    valid = np.isfinite(null)
    p = np.nan
    if np.isfinite(observed) and valid.all():
        hits = null <= observed + 1e-12 if structural is None else np.abs(null) >= abs(observed) - 1e-12
        p = float((1 + hits.sum()) / (len(null) + 1))
    return dict(
        rho=observed,
        p=p,
        valid_permutations=int(valid.sum()),
        status="ok" if np.isfinite(p) else "undefined correlation/permutation",
    )


def ranked_groups(returns, ids, size=5):
    """Choose fixed top/bottom groups with stable identity as the return tie-break."""
    if len(returns) < 2 * size or size < 2:
        raise ValueError("Insufficient reservoirs for disjoint groups")
    order = np.lexsort((ids, -np.asarray(returns)))
    return order[:size], order[-size:]


def group_masks(n, top, bottom):
    """Identify dependent TT, BB, and TB pairs on the common upper triangle."""
    a, b = np.triu_indices(n, 1)
    ta, tb = np.isin(a, top), np.isin(b, top)
    ba, bb = np.isin(a, bottom), np.isin(b, bottom)
    return dict(TT=ta & tb, BB=ba & bb, TB=(ta & bb) | (ba & tb))


def exact_group_test(matrix, top, bottom):
    """Enumerate every conditional top-label allocation, including observed ties."""
    selected = np.concatenate([top, bottom])
    values = matrix[np.ix_(selected, selected)]
    size = len(top)
    upper = np.triu_indices(size, 1)
    observed = float(values[size:, size:][upper].mean() - values[:size, :size][upper].mean())
    null = []
    for chosen in combinations(range(2 * size), size):
        a = np.asarray(chosen)
        b = np.setdiff1d(np.arange(2 * size), a)
        null.append(values[np.ix_(b, b)][upper].mean() - values[np.ix_(a, a)][upper].mean())
    null = np.asarray(null)
    tolerance = 8 * np.finfo(float).eps * max(1, abs(observed), np.max(abs(null)))
    masks = group_masks(len(matrix), top, bottom)
    summaries = {name: float(matrix[np.triu_indices(len(matrix), 1)][mask].mean()) for name, mask in masks.items()}
    return dict(
        **summaries, delta=observed, p=float(np.mean(null >= observed - tolerance)), allocations=len(null), status="ok"
    )


def seed_weights(count, seeds, seed=0):
    """Sample shared seed labels with replacement and keep their multiplicities."""
    draws = np.random.default_rng(seed).integers(seeds, size=(count, seeds))
    return np.eye(seeds)[draws].sum(axis=1) / seeds, draws


def policy_pair_tensor(matrix, systems, ids, seeds):
    """Arrange raw policy distances as (reservoir pair, source seed, target seed)."""
    if matrix.shape != (len(systems), len(systems)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid raw policy distance matrix")
    if not np.allclose(matrix, matrix.T) or np.any(matrix < -1e-8):
        raise ValueError("Raw policy distances must be symmetric and nonnegative")
    if systems.duplicated(["csv_idx", "policy_seed"]).any() or len(systems) != len(ids) * len(seeds):
        raise ValueError("Duplicate or incomplete raw policy identities")
    lookup = systems.reset_index().set_index(["csv_idx", "policy_seed"])["index"]
    try:
        positions = np.asarray([[lookup.loc[(int(i), int(s))] for s in seeds] for i in ids])
    except KeyError as error:
        raise ValueError("Missing raw policy identity") from error
    a, b = np.triu_indices(len(ids), 1)
    return matrix[positions[a, :, None], positions[b, None, :]]


def bootstrap_distances(tensor, weights):
    """Average Cartesian seed combinations with multiplicity on both axes."""
    return np.einsum("bs,bt,pst->bp", weights, weights, tensor, optimize=True)


def interval(values):
    """Return a conditional percentile interval only when >=95% of draws are valid."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    valid = int(finite.sum())
    low, high = np.quantile(values[finite], [0.025, 0.975]) if valid >= 0.95 * len(values) else (np.nan, np.nan)
    return dict(
        ci_low=float(low),
        ci_high=float(high),
        bootstrap_valid=valid,
        bootstrap_total=len(values),
        bootstrap_undefined=int((~finite).sum()),
    )


def regression_lines(x, distances, grid):
    """Evaluate descriptive OLS lines without independent-pair standard errors."""
    centered = x - x.mean()
    denominator = centered @ centered
    if denominator <= 0:
        return np.full((len(distances), len(grid)), np.nan)
    slopes = distances @ centered / denominator
    return distances.mean(axis=1)[:, None] + slopes[:, None] * (grid - x.mean())[None, :]


def structural_distances(selected, pool):
    """Fit z-scores and shrinkage covariance using only the valid candidate pool."""
    if not np.isfinite(selected).all() or not np.isfinite(pool).all() or len(pool) < 2:
        raise ValueError("Descriptor arrays must be finite with a nonempty candidate pool")
    z, reference = standardize(selected, pool), standardize(pool)
    covariance = LedoitWolf().fit(reference)
    if not np.isfinite(covariance.precision_).all():
        raise ValueError("Unusable shrinkage descriptor covariance")
    matrices = {
        "euclidean": squareform(pdist(z)),
        "mahalanobis": squareform(pdist(z, metric="mahalanobis", VI=covariance.precision_)),
    }
    return matrices, dict(
        mean=pool.mean(axis=0),
        scale=pool.std(axis=0),
        covariance=covariance.covariance_,
        precision=covariance.precision_,
        shrinkage=covariance.shrinkage_,
        condition=np.linalg.cond(covariance.covariance_),
        candidate_count=len(pool),
    )
