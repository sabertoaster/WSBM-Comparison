"""Stage 1-3 of the pipeline: sample the WSBM design space, describe each
reservoir with eight graph descriptors, then stratify and MaxMin-select a
representative subset for the RL experiments.

Selection is on STRUCTURE only. No dynamical quantity is used anywhere here --
that keeps selection statistically independent of the InputDSA outcome measure.

Run:  python select_reservoirs.py [n_per_stratum_pool]
Out:  selection/descriptors.csv   every sampled reservoir + its descriptors
      selection/selected.csv      the M=100 chosen, with the seed to rebuild them
      selection/selection.png     descriptor-space PCA with the picks marked
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, csv, itertools
import numpy as np
import igraph as ig
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from esn_models import WSBMESN, build_omega
from reservoirpy.nodes.wsbm_esn import WSBMESN, build_omega

N_RESERVOIR = 200
SIGMA = 0.1
STRATA = ("assortative", "disassortative", "core_periphery", "mixed", "null")
N_SELECT = 20  # per stratum -> M = 100
DESCRIPTORS = [
    "modularity",
    "participation",
    "clustering",
    "efficiency",
    "cv_strength",
    "cv_betweenness",
    "lambda_ratio",
    "frac_negative",
]


# ---------------------------------------------------------------- sampling ---


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


def sample_params(stratum, rng):
    """Draw one random point from the design space for a given stratum."""
    K = int(rng.choice([2, 3, 4, 6, 8, 12, 16]))
    p0 = float(np.exp(rng.uniform(np.log(0.03), np.log(0.30))))
    p = dict(
        stratum=stratum,
        K=K,
        p0=p0,
        centre_u=float(rng.uniform(0.0, 10.0)),
        sigma_ratio=float(np.exp(rng.uniform(np.log(1 / 8), np.log(8)))),
        rho=float(np.exp(rng.uniform(np.log(1 / 16), np.log(16)))),
        size_alpha=float(rng.choice([np.inf, 100.0, 30.0, 10.0, 3.0])),
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
            centre_u=float(rng.uniform(2.0, 10.0)),
            size_alpha=np.inf,
        )
    elif stratum == "assortative":
        p["contrast_u"] = float(rng.uniform(0.5, 8.0))
    elif stratum == "disassortative":
        p["contrast_u"] = -float(rng.uniform(0.5, 8.0))
    elif stratum == "core_periphery":
        p["contrast_u"] = float(rng.uniform(1.0, 8.0))
        p["f"] = float(rng.uniform(0.2, 0.8))
        p["core_fraction"] = float(rng.uniform(0.05, 0.40))
    elif stratum == "mixed":
        p["contrast_u"] = float(rng.uniform(1.0, 8.0))
    return p


def build(p):
    """Instantiate the WSBMESN described by a parameter dict."""
    K, cu, mu = p["K"], p["contrast_u"], p["centre_u"]
    hi, lo = (mu + cu) * SIGMA, (mu - cu) * SIGMA
    sr = p["sigma_ratio"]
    s_b = SIGMA * K / (sr + K - 1)
    S = np.full((K, K), s_b)
    np.fill_diagonal(S, sr * s_b)

    kw = dict(
        n_reservoir=N_RESERVOIR,
        n_communities=K,
        sigma=S,
        random_state=p["seed"],
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
        nc = max(1, int(round(N_RESERVOIR * p["core_fraction"])))
        per = np.repeat(
            np.arange(1, K), np.diff(np.linspace(0, N_RESERVOIR - nc, K).astype(int))
        )
        kw["assignments"] = np.concatenate([np.zeros(nc, dtype=int), per])
        kw.pop("size_alpha")
    elif p["stratum"] == "mixed":
        motif = "mixed"

    dm = "core_periphery" if p["stratum"] == "core_periphery" else "assortative"
    kw["connectivity"] = density_matrix(p["rho"], K, p["p0"], dm, p["f"])
    return WSBMESN(motif=motif, **kw)


# ------------------------------------------------------------ descriptors ---


def descriptors(W, z):
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
    g = ig.Graph.Weighted_Adjacency(
        (A > 0).astype(float).tolist(), mode="undirected", attr="weight", loops=False
    )
    out["clustering"] = float(
        np.nan_to_num(np.mean(g.transitivity_local_undirected(mode="zero")))
    )

    # 4. global efficiency, distance = 1/|w|
    with np.errstate(divide="ignore"):
        D = np.where(A > 0, 1.0 / np.where(A > 0, A, 1), np.inf)
    np.fill_diagonal(D, 0.0)
    sp = shortest_path(D, method="D", directed=False)
    with np.errstate(divide="ignore"):
        inv = np.where(sp > 0, 1.0 / sp, 0.0)
    out["efficiency"] = float(inv.sum() / (n * (n - 1)))

    # 5. CV of node strength
    out["cv_strength"] = (
        float(strength.std() / strength.mean()) if strength.mean() > 0 else 0.0
    )

    # 6. CV of betweenness centrality (igraph, C speed)
    src, dst = np.triu_indices(n, 1)
    keep = A[src, dst] > 0
    gb = ig.Graph(
        n=n, edges=list(zip(src[keep].tolist(), dst[keep].tolist())), directed=False
    )
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
    try:
        g = build(p)
        g.initialize(np.zeros((1, 1)))
        d = descriptors(g.W, g.community_assignments)
    except Exception as exc:  # skip pathological draws
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
        nxt = int(np.argmax(dmin))
        picked.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return picked


def main(pool_per_stratum=1000):
    from concurrent.futures import ProcessPoolExecutor

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selection")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)

    params = [sample_params(s, rng) for s in STRATA for _ in range(pool_per_stratum)]
    print(f"sampling {len(params)} reservoirs (N={N_RESERVOIR}) ...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(describe_one, params, chunksize=8), 1):
            if r is not None:
                rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(params)}  kept {len(rows)}", flush=True)
    print(f"described {len(rows)} reservoirs ({len(params)-len(rows)} rejected)")

    keys = list(rows[0].keys())
    with open(os.path.join(outdir, "descriptors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # standardise on the FULL pool so strata share one descriptor space
    D = np.array([[r[k] for k in DESCRIPTORS] for r in rows])
    Z = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1.0)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    PCs = U * S
    evr = S**2 / (S**2).sum()
    print(
        "PCA explained variance:",
        np.round(evr[:5], 3),
        f"(first 3 = {evr[:3].sum():.1%})",
    )

    selected = []
    for s in STRATA:
        idx = np.array([i for i, r in enumerate(rows) if r["stratum"] == s])
        pick = maxmin(PCs[idx][:, :5], N_SELECT, rng)
        selected += [int(idx[j]) for j in pick]
        print(f"  {s:<16} pool {len(idx):>5} -> selected {len(pick)}")

    with open(os.path.join(outdir, "selected.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for i in selected:
            w.writerow(rows[i])
    np.save(os.path.join(outdir, "pcs.npy"), PCs)

    _plot(outdir, PCs, rows, selected, evr)
    print(f"\nwrote {len(selected)} selected reservoirs to {outdir}")
    return rows, selected


def _plot(outdir, PCs, rows, selected, evr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strat = np.array([r["stratum"] for r in rows])
    sel = np.zeros(len(rows), bool)
    sel[selected] = True
    cols = dict(zip(STRATA, ["#c0392b", "#1f6fb4", "#f39c12", "#7d3c98", "#555555"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2)]):
        for s in STRATA:
            m = strat == s
            ax.scatter(PCs[m, a], PCs[m, b], s=5, c=cols[s], alpha=0.18, lw=0)
            ax.scatter(
                PCs[m & sel, a],
                PCs[m & sel, b],
                s=42,
                c=cols[s],
                edgecolors="k",
                linewidths=0.6,
                label=s,
            )
        ax.set_xlabel(f"PC{a+1} ({evr[a]:.0%})")
        ax.set_ylabel(f"PC{b+1} ({evr[b]:.0%})")
    axes[0].legend(fontsize=8, frameon=False, markerscale=0.8)
    fig.suptitle(
        "Descriptor space: full pool (faint) and MaxMin selection (outlined)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "selection.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
