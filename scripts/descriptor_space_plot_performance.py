import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Import get_performance from the same directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from get_top_bottom_10_3_tasks import get_performance
except ImportError:
    print("Error: Could not import get_performance from get_top_bottom_10_3_tasks.py")
    sys.exit(1)

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

STRATA_SHAPES = {
    "assortative": "s",       # square
    "disassortative": "o",    # circle
    "core_periphery": "^",    # triangle
    "mixed": "*",             # star
    "null": "p"               # pentagon
}

def main():
    parser = argparse.ArgumentParser(description="Plot performance of 100 reservoirs in PCA descriptor space.")
    parser.add_argument("task", type=str, help="RL Task name (e.g., Swimmer-v4)")
    args = parser.parse_args()

    task = args.task

    # Data loading
    pool_csv = os.path.join(os.path.dirname(__file__), "selection", "descriptors.csv")
    if not os.path.exists(pool_csv):
        print(f"Error: Full pool file not found at {pool_csv}")
        sys.exit(1)

    selected_csv = os.path.join(os.path.dirname(__file__), "..", "selected.csv")
    if not os.path.exists(selected_csv):
        selected_csv = os.path.join(os.path.dirname(__file__), "selection", "selected.csv")
        if not os.path.exists(selected_csv):
            print(f"Error: Could not find selected.csv")
            sys.exit(1)

    print(f"Loading full pool from {pool_csv}...")
    df_pool = pd.read_csv(pool_csv)
    df_pool["stratum"] = df_pool["stratum"].fillna("null")

    print(f"Loading selected reservoirs from {selected_csv}...")
    df_sel = pd.read_csv(selected_csv)
    df_sel["stratum"] = df_sel["stratum"].fillna("null")

    # PCA on full pool
    D_pool = df_pool[DESCRIPTORS].values
    mean_pool = D_pool.mean(axis=0)
    std_pool = np.where(D_pool.std(axis=0) > 0, D_pool.std(axis=0), 1.0)

    Z_pool = (D_pool - mean_pool) / std_pool
    # Center Z before SVD
    Z_pool_centered = Z_pool - Z_pool.mean(axis=0)

    U, S, Vt = np.linalg.svd(Z_pool_centered, full_matrices=False)
    PCs_pool = U * S
    evr = S**2 / (S**2).sum()
    print("PCA explained variance:", np.round(evr[:5], 3))

    # Project selected onto PCs
    D_sel = df_sel[DESCRIPTORS].values
    Z_sel = (D_sel - mean_pool) / std_pool
    # Projection: Z_sel_centered @ Vt.T
    Z_sel_centered = Z_sel - Z_pool.mean(axis=0)
    PCs_sel = Z_sel_centered @ Vt.T

    # Load performance
    print(f"Fetching performance for {task}...")
    # Assume 10000 window steps, linear weighting as default in other scripts
    results = get_performance(task, window_steps=10000, weight_type="linear")

    # Create mapping from seed_name to performance
    perf_map = {res["seed_name"]: res["performance"] for res in results}

    performances = []
    found_count = 0

    # We assume the index in selected.csv matches idx in the seed_name
    for idx, row in df_sel.iterrows():
        motif = row["stratum"]
        seed_name = f"{task}_RES_{motif}_idx{idx}_PPO_500000steps"

        if seed_name in perf_map:
            performances.append(perf_map[seed_name])
            found_count += 1
        else:
            print(f"Warning: Missing performance for {seed_name}")
            performances.append(np.nan)

    df_sel["performance"] = performances
    print(f"Found performance for {found_count}/{len(df_sel)} reservoirs.")

    if found_count == 0:
        print("Error: No performance records found. Aborting plot.")
        sys.exit(1)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # Normalize performance for colormap
    valid_perf = df_sel["performance"].dropna().values
    norm = Normalize(vmin=valid_perf.min(), vmax=valid_perf.max())
    cmap = plt.cm.viridis

    for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
        # 1. Plot full pool faint grey
        ax.scatter(
            PCs_pool[:, pc_x], PCs_pool[:, pc_y],
            s=5, color="#dddddd", alpha=0.2, lw=0, zorder=1
        )

        # 2. Plot selected reservoirs
        for stratum, shape in STRATA_SHAPES.items():
            mask = (df_sel["stratum"] == stratum) & df_sel["performance"].notna()
            if not mask.any():
                continue

            ax.scatter(
                PCs_sel[mask, pc_x],
                PCs_sel[mask, pc_y],
                c=df_sel.loc[mask, "performance"],
                cmap=cmap,
                norm=norm,
                s=60,
                marker=shape,
                edgecolors="k",
                linewidths=0.8,
                zorder=2,
                label=stratum
            )

        ax.set_xlabel(f"PC{pc_x+1} ({evr[pc_x]:.0%})")
        ax.set_ylabel(f"PC{pc_y+1} ({evr[pc_y]:.0%})")

    # Legend for shapes
    handles = []
    for stratum, shape in STRATA_SHAPES.items():
        if (df_sel["stratum"] == stratum).any():
            # Create dummy lines for legend
            h = mlines.Line2D(
                [], [], color='none', marker=shape, markersize=8,
                markeredgecolor='k', markerfacecolor='gray', label=stratum
            )
            handles.append(h)
    axes[0].legend(handles=handles, fontsize=9, frameon=True, facecolor='white', framealpha=1.0, loc="best")

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("Performance")

    fig.suptitle(
        f"Descriptor Space: Full Pool (faint) & Selected (outlined) colored by {task} Performance",
        fontsize=12
    )

    out_file = f"inputdsa_100_reservoirs_performance_PCA_{task}.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
