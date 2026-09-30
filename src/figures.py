"""Headless research figures saved inside isolated run directories."""

import numpy as np
from scipy.spatial.distance import pdist

from src.utils import DESCRIPTORS, STRATA, align_curves, read_scalars, standardize

COLORS = dict(zip(STRATA, ("tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")))


def pyplot():
    """Load a headless pyplot backend only when a figure is requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_distances(matrices, labels, path, dpi=150):
    """Plot the five distance matrices with identical row and column ordering."""
    plt = pyplot()
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5), squeeze=False)
    for axis, (name, matrix) in zip(axes[0], matrices.items()):
        image = axis.imshow(matrix, cmap="viridis")
        fig.colorbar(image, ax=axis, shrink=0.7)
        axis.set_title(name.replace("_", " "))
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_structural(frame, matrix, path, reference=None, dpi=150):
    """Plot descriptor versus state distance and return an exportable pair table."""
    import pandas as pd

    plt = pyplot()
    descriptors = standardize(frame[list(DESCRIPTORS)].to_numpy(), reference)
    structural = pdist(descriptors)
    upper = np.triu_indices(len(frame), 1)
    dynamic = matrix[upper]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(structural, dynamic, s=12, alpha=0.2)
    image = axes[1].hexbin(structural, dynamic, gridsize=25, mincnt=1, cmap="viridis")
    fig.colorbar(image, ax=axes[1], label="Reservoir pairs")
    if len(structural) > 1 and structural.std() > 0:
        coefficients = np.polyfit(structural, dynamic, 1)
        x = np.array([structural.min(), structural.max()])
        for axis in axes:
            axis.plot(x, np.polyval(coefficients, x), color="tab:red")
    for axis in axes:
        axis.set_xlabel("Structural distance (standardized 8D Euclidean)")
        axis.set_ylabel("State distance (joint alignment)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    ids = frame.csv_idx.to_numpy()
    return pd.DataFrame(
        {
            "csv_idx_a": ids[upper[0]],
            "csv_idx_b": ids[upper[1]],
            "structural_distance": structural,
            "state_distance": dynamic,
        }
    )


def plot_selection(pool, selected, pcs, explained, path, dpi=150):
    """Plot the full structural pool and chosen reservoirs in PCA coordinates."""
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    selected = set(selected)
    for axis, (a, b) in zip(axes, ((0, 1), (0, 2))):
        if b >= pcs.shape[1]:
            axis.set_visible(False)
            continue
        for stratum in STRATA:
            mask = pool.stratum.to_numpy() == stratum
            chosen = mask & np.array([i in selected for i in range(len(pool))])
            axis.scatter(pcs[mask, a], pcs[mask, b], s=5, color=COLORS[stratum], alpha=0.2)
            axis.scatter(pcs[chosen, a], pcs[chosen, b], s=35, color=COLORS[stratum], edgecolors="k", label=stratum)
        axis.set_xlabel(f"PC{a + 1} ({explained[a]:.1%})")
        axis.set_ylabel(f"PC{b + 1} ({explained[b]:.1%})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_performance_pca(pool, frame, path, dpi=150):
    """Project selected descriptors using the full pool PCA and color by return."""
    plt = pyplot()
    full = pool[list(DESCRIPTORS)].to_numpy()
    standardized = standardize(full)
    _, singular, rotation = np.linalg.svd(standardized, full_matrices=False)
    pcs = standardized @ rotation.T
    selected = standardize(frame[list(DESCRIPTORS)].to_numpy(), full) @ rotation.T
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = None
    for axis, b in zip(axes, (1, 2)):
        if b >= pcs.shape[1]:
            axis.set_visible(False)
            continue
        axis.scatter(pcs[:, 0], pcs[:, b], s=4, color="gray", alpha=0.15)
        scatter = axis.scatter(selected[:, 0], selected[:, b], c=frame.performance, cmap="viridis", edgecolors="k")
        axis.set_xlabel("PC1")
        axis.set_ylabel(f"PC{b + 1}")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Mean evaluation return / legacy reward score")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training(registry, path, dpi=150, aggregate=False):
    """Plot learning curves, aligning policy replicates by actual steps."""
    plt = pyplot()
    tasks = registry.task.unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        for stratum, records in registry[registry.task == task].groupby("stratum"):
            color = COLORS.get(stratum, "black")
            curves = [read_scalars(record.log_path) for _, record in records.iterrows()]
            if aggregate:
                steps, mean, std = align_curves(curves)
                axis.plot(steps, mean, color=color, label=stratum)
                axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
            else:
                for index, (steps, values) in enumerate(curves):
                    axis.plot(
                        steps,
                        values,
                        color=color,
                        alpha=1 if stratum == "PPO" else 0.3,
                        label=stratum if index == 0 else None,
                    )
        axis.set_title(task)
        axis.set_xlabel("Training steps")
        axis.set_ylabel("Mean episode reward")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def generate_motifs(args, output):
    """Draw four canonical motifs with a shared edge-weight color scale."""
    import networkx as nx

    from src.reservoirs import build_corrected

    plt = pyplot()
    matrices, assignments = [], []
    if args.protocol == "legacy":
        from src.wsbm_esn import AssortativeESN, CorePeripheryESN, DisassortativeESN, MixedESN

        specifications = (
            (AssortativeESN, {"mu_in": args.figure_hi, "mu_out": args.figure_lo}),
            (DisassortativeESN, {"mu_in": args.figure_lo, "mu_out": args.figure_hi}),
            (
                CorePeripheryESN,
                {
                    "mu_cc": args.figure_hi,
                    "mu_cp": args.figure_mid,
                    "mu_pp": args.figure_lo,
                    "core_fraction": args.figure_core_fraction,
                },
            ),
            (MixedESN, {"hi": args.figure_hi, "lo": args.figure_lo}),
        )
        for constructor, settings in specifications:
            esn = constructor(
                n_reservoir=args.units,
                n_communities=args.n_communities,
                random_state=args.figure_seed,
                sigma=args.sigma_base,
                spectral_radius=args.res_sr,
                leak_rate=args.res_lr,
                input_scaling=args.res_iss,
                connectivity=args.figure_connectivity,
                **settings,
            )
            matrices.append(esn.W_res)
            assignments.append(esn.community_assignments)
    else:
        for motif in STRATA[:4]:
            artifact = build_corrected(
                {
                    "stratum": motif,
                    "seed": args.figure_seed,
                    "K": args.n_communities,
                    "size_alpha": np.inf,
                    "core_fraction": args.figure_core_fraction,
                    "centre_u": (args.figure_hi + args.figure_lo) / (2 * args.sigma_base),
                    "contrast_u": (args.figure_hi - args.figure_lo)
                    / (2 * args.sigma_base)
                    * (-1 if motif == "disassortative" else 1),
                    "sigma_ratio": 1,
                    "rho": 1,
                    "p0": args.figure_connectivity,
                    "f": (args.figure_mid - args.figure_lo) / (args.figure_hi - args.figure_lo),
                },
                args.units,
                args.sigma_base,
                args.res_sr,
            )
            matrices.append(artifact["W"])
            assignments.append(artifact["assignments"])
    maximum = max(float(np.abs(matrix).max()) for matrix in matrices)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for axis, motif, matrix, assignment in zip(axes.flat, STRATA[:4], matrices, assignments):
        if args.figure_layout == "community":
            from types import SimpleNamespace

            esn = SimpleNamespace(
                W_res=matrix, n_reservoir=len(matrix), get_community_data=lambda: {"community_assignments": assignment}
            )
            visualize_esn(esn, motif.replace("_", " "), ax=axis, show=False, w_min=0, w_max=maximum)
            continue
        graph = nx.from_numpy_array(matrix)
        layout = nx.spring_layout(graph, seed=args.figure_seed)
        edges = list(graph.edges())
        weights = [abs(matrix[a, b]) for a, b in edges]
        nx.draw_networkx(
            graph,
            pos=layout,
            ax=axis,
            node_color=assignment,
            cmap="tab20",
            node_size=25,
            with_labels=False,
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0,
            edge_vmax=maximum,
            width=0.3,
        )
        axis.set_title(motif.replace("_", " "))
        axis.set_axis_off()
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, maximum), cmap="viridis"),
        ax=list(axes.flat),
        label="Absolute connection strength",
        shrink=0.6,
    )
    fig.savefig(output / "figures" / "wsbm_motifs.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _get_node_to_community(esn):
    """Return node/community labels from either supported ESN implementation."""
    if hasattr(esn, "get_community_data"):
        community_data = esn.get_community_data()
        assignments = community_data.get("community_assignments")
        if assignments is not None:
            return {i: int(c) for i, c in enumerate(assignments)}
    if hasattr(esn, "get_node_metadata"):
        node_metadata = esn.get_node_metadata()
        is_core = node_metadata.get("is_core")
        if is_core is not None:
            return {i: int(c) for i, c in enumerate(is_core)}
    return {i: 0 for i in range(esn.n_reservoir)}


def visualize_esn(
    esn,
    title: str,
    ax=None,
    show: bool = True,
    title_size: int = 12,
    param_text=None,
    w_min=None,
    w_max=None,
):
    """Draw a community layout with bundled edges; optionally display interactively.

    Accepts the historical ESN interface and optional shared weight bounds.
    Returns the minimum and maximum absolute edge weights used for coloring.
    """
    import networkx as nx
    from netgraph import Graph

    plt = pyplot()
    adj_matrix = np.array(esn.W_res)
    res_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    node_to_community = _get_node_to_community(esn)
    community_ids = sorted(set(node_to_community.values()))
    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    community_to_color = {community_id: palette[i % len(palette)] for i, community_id in enumerate(community_ids)}
    node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}
    edges = list(res_graph.edges())

    # Calculate global weights if not provided
    if w_min is None or w_max is None:
        if edges:
            weights = np.array([abs(adj_matrix[u, v]) for u, v in edges], dtype=float)
            w_min = weights.min() if w_min is None else w_min
            w_max = weights.max() if w_max is None else w_max
        else:
            w_min, w_max = 0.0, 1.0

    if edges:
        weights = np.array([abs(adj_matrix[u, v]) for u, v in edges], dtype=float)
        min_alpha, max_alpha = 0.05, 1.0
        if np.isclose(w_min, w_max):
            edge_alpha = {edge: max_alpha for edge in edges}
            edge_color = {edge: plt.cm.viridis(1.0) for edge in edges}
        else:
            scaled = np.clip((weights - w_min) / (w_max - w_min), 0.0, 1.0)
            edge_alpha = {edge: float(min_alpha + s * (max_alpha - min_alpha)) for edge, s in zip(edges, scaled)}
            edge_color = {edge: plt.cm.viridis(s) for edge, s in zip(edges, scaled)}
    else:
        edge_alpha = 1.0
        edge_color = "k"

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    Graph(
        res_graph,
        node_color=node_color,
        node_edge_width=0,
        edge_alpha=edge_alpha,
        edge_color=edge_color,
        node_layout="community",
        node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout="bundled",
        edge_layout_kwargs=dict(k=2000),
    )
    if param_text:
        ax.text(
            0.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
    ax.set_title(title, fontsize=title_size)
    ax.set_axis_off()
    if show:
        plt.show()
    return w_min, w_max
