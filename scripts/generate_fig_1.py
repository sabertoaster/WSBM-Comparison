import sys
import os


from src.wsbm_esn import AssortativeESN, DisassortativeESN, CorePeripheryESN, MixedESN
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import Graph

assortative_esn = AssortativeESN(n_communities=6, random_state=42)
disassortative_esn = DisassortativeESN(n_communities=6, random_state=42)
core_esn = CorePeripheryESN(n_communities=6, random_state=42)
mix_esn = MixedESN(n_communities=6, random_state=42)


def _get_node_to_community(esn):
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
):
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
    community_to_color = {
        community_id: palette[i % len(palette)]
        for i, community_id in enumerate(community_ids)
    }
    node_color = {
        node: community_to_color[community_id]
        for node, community_id in node_to_community.items()
    }
    edges = list(res_graph.edges())
    if edges:
        weights = np.array([abs(adj_matrix[u, v]) for u, v in edges], dtype=float)
        w_min = weights.min()
        w_max = weights.max()
        min_alpha, max_alpha = 0.05, 1.0
        if np.isclose(w_min, w_max):
            edge_alpha = {edge: max_alpha for edge in edges}
        else:
            scaled = (weights - w_min) / (w_max - w_min)
            edge_alpha = {
                edge: float(min_alpha + s * (max_alpha - min_alpha))
                for edge, s in zip(edges, scaled)
            }
    else:
        edge_alpha = 1.0
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    Graph(
        res_graph,
        node_color=node_color,
        node_edge_width=0,
        edge_alpha=edge_alpha,
        node_layout="community",
        node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout="bundled",
        edge_layout_kwargs=dict(k=2000),
        edge_cmap=plt.cm.viridis,
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


# visualize_esn(assortative_esn, "AssortativeESN Reservoir Graph")
# visualize_esn(disassortative_esn, "DisassortativeESN Reservoir Graph")
# visualize_esn(core_esn, "CorePeripheryESN Reservoir Graph")
# visualize_esn(mix_esn, "MixedESN Reservoir Graph")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

visualize_esn(
    assortative_esn, "Assortative ESN", ax=axes[0, 0], show=False, title_size=16
)
visualize_esn(
    disassortative_esn, "Disassortative ESN", ax=axes[0, 1], show=False, title_size=16
)
visualize_esn(core_esn, "Core-Periphery ESN", ax=axes[1, 0], show=False, title_size=16)
visualize_esn(mix_esn, "Mixed ESN", ax=axes[1, 1], show=False, title_size=16)

plt.tight_layout()
plt.savefig("myplot.png")
