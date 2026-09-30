"""Structural sampling and descriptor calculation, independent of RL outcomes.

Legacy builders retain the historical selection equations. Corrected selection
uses the canonical factory and persists the realized recurrent matrix.
"""

import igraph as ig
import numpy as np
from reservoirpy.nodes.wsbm_esn import WSBMESN
from scipy.sparse.csgraph import shortest_path

N_RESERVOIR = 200
SIGMA = 0.1


def density_matrix(rho, K, p0, motif="assortative", f=0.5):
    """(K,K) edge probabilities with mean density pinned at p0.

    Solving (1/K)p_w + ((K-1)/K)p_b == p0 keeps the expected edge COUNT fixed
    across rho, so a topology motif is never confounded with edge abundance.
    """
    if K == 1:
        return np.full((1, 1), p0)
    p_b = p0 * K / (rho + K - 1)
    p_w = rho * p_b
    if motif == "core_periphery":
        P = np.full((K, K), p_b)
        P[0, :] = P[:, 0] = p_b + f * (p_w - p_b)
        P[0, 0] = p_w
    else:
        P = np.full((K, K), p_b)
        np.fill_diagonal(P, p_w)
    return np.clip(P, 0.0, 1.0)


def sample_params(
    stratum,
    rng,
    communities=(2, 3, 4, 6, 8, 12, 16),
    density_range=(0.03, 0.30),
    centre_range=(0.0, 10.0),
    dispersion_range=(1 / 8, 8),
    topology_range=(1 / 16, 16),
    size_alphas=(np.inf, 100, 30, 10, 3),
    contrast_range=(0.5, 8),
    core_range=(0.05, 0.40),
    mixing_range=(0.2, 0.8),
    null_centre_range=(2.0, 10.0),
    core_contrast_range=(1.0, 8.0),
    mixed_contrast_range=(1.0, 8.0),
):
    """Draw one random point from the design space for a given stratum."""
    K = int(rng.choice(communities))
    p0 = float(np.exp(rng.uniform(*np.log(density_range))))
    p = dict(
        stratum=stratum,
        K=K,
        p0=p0,
        centre_u=float(rng.uniform(*centre_range)),
        sigma_ratio=float(np.exp(rng.uniform(*np.log(dispersion_range)))),
        rho=float(np.exp(rng.uniform(*np.log(topology_range)))),
        size_alpha=float(rng.choice(size_alphas)),
        f=0.5,
        core_fraction=0.0,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    if stratum == "null":
        # the Erdos-Renyi arm: no weight contrast, no topology contrast,
        # no dispersion contrast. Only density and seed vary.
        p.update(
            contrast_u=0.0,
            rho=1.0,
            sigma_ratio=1.0,
            centre_u=float(rng.uniform(*null_centre_range)),
            size_alpha=np.inf,
        )
    elif stratum == "assortative":
        p["contrast_u"] = float(rng.uniform(*contrast_range))
    elif stratum == "disassortative":
        p["contrast_u"] = -float(rng.uniform(*contrast_range))
    elif stratum == "core_periphery":
        p["contrast_u"] = float(rng.uniform(*core_contrast_range))
        p["f"] = float(rng.uniform(*mixing_range))
        p["core_fraction"] = float(rng.uniform(*core_range))
    elif stratum == "mixed":
        p["contrast_u"] = float(rng.uniform(*mixed_contrast_range))
    return p


def build(p, units=N_RESERVOIR, sigma_base=SIGMA, sr=0.9):
    """Instantiate the WSBMESN described by a parameter dict."""
    K, cu, mu = p["K"], p["contrast_u"], p["centre_u"]
    hi, lo = (mu + cu) * sigma_base, (mu - cu) * sigma_base
    dispersion = p["sigma_ratio"]
    s_b = sigma_base * K / (dispersion + K - 1)
    S = np.full((K, K), s_b)
    np.fill_diagonal(S, dispersion * s_b)

    kw = dict(
        n_reservoir=units,
        n_communities=K,
        sigma=S,
        random_state=p["seed"],
        spectral_radius=sr,
        hi=hi,
        lo=lo,
        size_alpha=None if not np.isfinite(p["size_alpha"]) else p["size_alpha"],
    )

    motif = "assortative"
    if p["stratum"] == "disassortative":
        motif = "assortative"  # negative contrast IS disassortative
    elif p["stratum"] == "core_periphery":
        motif = "core_periphery"
        kw["mid"] = lo + p["f"] * (hi - lo)
        nc = max(1, int(round(units * p["core_fraction"])))
        per = np.repeat(np.arange(1, K), np.diff(np.linspace(0, units - nc, K).astype(int)))
        kw["assignments"] = np.concatenate([np.zeros(nc, dtype=int), per])
        kw.pop("size_alpha")
    elif p["stratum"] == "mixed":
        motif = "mixed"

    dm = "core_periphery" if p["stratum"] == "core_periphery" else "assortative"
    kw["connectivity"] = density_matrix(p["rho"], K, p["p0"], dm, p["f"])
    return WSBMESN(motif=motif, **kw)


# ------------------------------------------------------------ descriptors ---


def descriptors(W, z, weighted=False):
    """Eight structural descriptors.

    Topology/strength measures use |W| (distances and modularity are undefined
    for signed weights); the spectral ratio and the negative fraction use the
    signed matrix, and are the only two that see the sign structure.
    """
    A = np.abs(W)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    strength = A.sum(1)
    m2 = A.sum()
    out = {}

    # 1. weighted modularity of the PLANTED partition
    if m2 > 0:
        same = z[:, None] == z[None, :]
        expect = np.outer(strength, strength) / m2
        out["modularity"] = float(((A - expect) * same).sum() / m2)
    else:
        out["modularity"] = 0.0

    # 2. mean participation coefficient
    K = int(z.max()) + 1
    kis = np.zeros((n, K))
    for s in range(K):
        kis[:, s] = A[:, z == s].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(strength[:, None] > 0, kis / strength[:, None], 0.0)
    out["participation"] = float((1.0 - (frac**2).sum(1)).mean())

    # 3. weighted clustering coefficient (Onnela), via igraph on |W|
    g = ig.Graph.Weighted_Adjacency((A > 0).astype(float).tolist(), mode="undirected", attr="weight", loops=False)
    out["clustering"] = float(np.nan_to_num(np.mean(g.transitivity_local_undirected(mode="zero"))))

    if weighted:
        # Onnela: geometric mean of normalized edge weights in each triangle.
        normalized = A / A.max() if A.max() > 0 else A
        cube = np.cbrt(normalized)
        degree = (A > 0).sum(axis=1)
        triangles = np.diag(cube @ cube @ cube)
        denom = degree * (degree - 1)
        out["clustering"] = float(np.divide(triangles, denom, out=np.zeros(n), where=denom > 0).mean())

    # 4. global efficiency, distance = 1/|w|
    with np.errstate(divide="ignore"):
        D = np.where(A > 0, 1.0 / np.where(A > 0, A, 1), np.inf)
    np.fill_diagonal(D, 0.0)
    sp = shortest_path(D, method="D", directed=False)
    with np.errstate(divide="ignore"):
        inv = np.where(sp > 0, 1.0 / sp, 0.0)
    out["efficiency"] = float(inv.sum() / (n * (n - 1)))

    # 5. CV of node strength
    out["cv_strength"] = float(strength.std() / strength.mean()) if strength.mean() > 0 else 0.0

    # 6. CV of betweenness centrality (igraph, C speed)
    src, dst = np.triu_indices(n, 1)
    keep = A[src, dst] > 0
    gb = ig.Graph(n=n, edges=list(zip(src[keep].tolist(), dst[keep].tolist())), directed=False)
    if gb.ecount():
        gb.es["weight"] = (1.0 / A[src[keep], dst[keep]]).tolist()
        b = np.asarray(gb.betweenness(weights="weight"), dtype=float)
        out["cv_betweenness"] = float(b.std() / b.mean()) if b.mean() > 0 else 0.0
    else:
        out["cv_betweenness"] = 0.0

    # 7. spectral ratio |lambda_2 / lambda_1|  (signed W, symmetric -> real)
    ev = np.sort(np.abs(np.linalg.eigvalsh(W)))
    out["lambda_ratio"] = float(ev[-2] / ev[-1]) if ev[-1] > 0 else 0.0

    # 8. fraction of existing edges that are inhibitory (signed W)
    nz = W[W != 0]
    out["frac_negative"] = float((nz < 0).mean()) if nz.size else 0.0
    return out


def describe_one(p):
    """Return sampled parameters plus finite structural descriptors, or None for a rejected draw."""
    try:
        g = build(p)
        g.initialize(np.zeros((1, 1)))
        d = descriptors(g.W, g.community_assignments)
    except Exception:  # skip pathological draws
        return None
    if not all(np.isfinite(v) for v in d.values()):
        return None
    row = dict(p)
    row.update(d)
    return row


# --------------------------------------------------------------- selection ---


def maxmin(X, k, rng):
    """MaxMin (Kennard-Stone) diversity selection in descriptor space."""
    n = len(X)
    if n <= k:
        return list(range(n))
    first = int(np.argmax(((X - X.mean(0)) ** 2).sum(1)))  # most extreme point
    picked = [first]
    dmin = np.linalg.norm(X - X[first], axis=1)
    for _ in range(k - 1):
        dmin[picked] = -np.inf
        nxt = int(np.argmax(dmin))
        picked.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return picked
