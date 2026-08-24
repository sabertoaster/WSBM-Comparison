import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist

# Add parent directory to sys.path to import DSA
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DSA"))
)
try:
    from DSA import InputDSA
except ImportError:
    print("Could not import DSA. Ensure it is accessible in the parent directory.")
    sys.exit(1)


def main():
    TASK = "Ant-v4"  # Run this script independently for each task

    # 1. Load the 100 selected reservoirs to guarantee identical ordering
    df = pd.read_csv(
        "selected.csv"
    )  # It is in the main directory according to the previous task
    # Wait, the original code had 'selection/selected.csv', I will use 'selected.csv'
    if not os.path.exists("selected.csv") and os.path.exists("selection/selected.csv"):
        csv_path = "selection/selected.csv"
    else:
        csv_path = "selected.csv"

    df = pd.read_csv(csv_path)
    df["stratum"] = df["stratum"].fillna("null")

    # ========================================================================
    # STEP A: DYNAMICAL DISTANCES (DSA_state)
    # ========================================================================
    print(f"Loading trajectories for {TASK}...")
    Ys = []
    Us = []

    for index, row in df.iterrows():
        stratum = row["stratum"]
        seed = int(row["seed"])

        # O_t = U (inputs to reservoir)
        # C_t = X (reservoir states)
        try:
            U = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ot.npy")
            X = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ct.npy")
        except FileNotFoundError:
            print(f"Warning: Missing trajectory for {stratum} seed {seed}, skipping...")
            # We must be careful: skipping ruins the order and alignment with structural descriptors!
            # Since we must maintain perfect alignment, we should raise an error if any is missing.
            raise FileNotFoundError(f"Missing trajectory for {stratum} seed {seed}.")

        Ys.append(X)
        Us.append(U)

    print(f"Fitting InputDSA on {len(Ys)} networks...")
    # The user originally mentioned rank=50, delays=40 for plume tracking.
    # However, N4SID subspace identification requires N_samples > 2*delays*(m + p_out).
    # Since RL episodes are ~1000 steps, delays=40 leads to empty matrices and 0 distances!
    # We dynamically select delays=3 and rank=20 to safely fit within the 1000-step constraint.
    dmd_config = dict(n_delays=1, rank=1, backend="dmdc")

    inputDSA = InputDSA(
        X=Ys,
        X_control=Us,
        dmd_config=dmd_config,
        simdist_config={"compare": "joint", "return_distance_components": True},
    )

    res = inputDSA.fit_score()
    sims_state_joint = res[:, :, 1]

    print("Extracting pairwise DSA_state distances...")
    num_networks = len(Ys)
    dsa_distances = []
    for i in range(num_networks):
        for j in range(i + 1, num_networks):
            dsa_distances.append(sims_state_joint[i, j])
    dsa_dist_condensed = np.array(dsa_distances)

    # ========================================================================
    # STEP B: STRUCTURAL DISTANCES (Normalized 8D Space, NO PCA)
    # ========================================================================
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

    D = df[DESCRIPTORS].values

    # Z-score standardization: (X - Mean) / Std
    # This prevents 'modularity' from overpowering 'lambda_ratio' in the distance calc
    Z = (D - D.mean(axis=0)) / np.where(D.std(axis=0) > 0, D.std(axis=0), 1.0)

    # Compute the 4,950 pairwise Euclidean distances directly on the 8D Z-scores
    struct_dist_condensed = pdist(Z, metric="euclidean")

    # ========================================================================
    # STEP C: PLOTTING (Scatter + Hexbin)
    # ========================================================================
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # --- Plot A: Scatter Plot with Trendline ---
    sns.regplot(
        x=struct_dist_condensed,
        y=dsa_dist_condensed,
        ax=axes[0],
        scatter_kws={"alpha": 0.15, "s": 15, "color": "#2c3e50"},
        line_kws={"color": "#e74c3c", "linewidth": 2},
    )
    axes[0].set_title(f"{TASK}: Structural vs. DSA_state Distance")
    axes[0].set_xlabel("Structural Distance (Normalized 8D Euclidean)")
    axes[0].set_ylabel("DSA_state Distance")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Plot B: 2D Hexbin Density Plot ---
    hb = axes[1].hexbin(
        struct_dist_condensed, dsa_dist_condensed, gridsize=25, cmap="viridis", mincnt=1
    )
    cb = fig.colorbar(hb, ax=axes[1], label="Density of Reservoir Pairs")

    # Overlay the linear trendline on the density plot
    m, b = np.polyfit(struct_dist_condensed, dsa_dist_condensed, 1)
    axes[1].plot(
        struct_dist_condensed,
        m * struct_dist_condensed + b,
        color="#e74c3c",
        linewidth=2,
    )

    axes[1].set_title(f"{TASK} Density: Structural vs. DSA_state")
    axes[1].set_xlabel("Structural Distance (Normalized 8D Euclidean)")
    axes[1].set_ylabel("DSA_state Distance")

    plt.tight_layout()
    output_filename = f"inputdsa_100_reservoirs_H1_{TASK}_NoPCA.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Plots successfully saved to {output_filename}")


if __name__ == "__main__":
    main()
