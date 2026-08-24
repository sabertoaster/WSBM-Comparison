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

# Import the get_performance function to fetch the rankings
try:
    from get_top_bottom_10_3_tasks import get_performance
except ImportError:
    # Handle if scripts isn't in path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from get_top_bottom_10_3_tasks import get_performance

try:
    from src.utils import choose_rank
except ImportError:
    # Add root dir to path if not present
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils import choose_rank


def main():
    TASK = "Walker2d-v4"  # Run this script independently for each task

    # 1. Load the 100 selected reservoirs
    if not os.path.exists("selected.csv") and os.path.exists("selection/selected.csv"):
        csv_path = "selection/selected.csv"
    else:
        csv_path = "selected.csv"

    df_full = pd.read_csv(csv_path)
    df_full["stratum"] = df_full["stratum"].fillna("null")

    # 2. Get top 10 and bottom 10 reservoirs
    print(f"Calculating performance rankings for {TASK}...")
    results = get_performance(TASK, window_steps=10000, weight_type="linear")

    # Generate the expected directory name (seed_name) for each row in df
    # so we can match them with the results
    valid_seed_names = []
    for idx, row in df_full.iterrows():
        motif = row["stratum"]
        valid_seed_names.append(f"{TASK}_RES_{motif}_idx{idx}_PPO_500000steps")

    # Filter results to only include those in our selected.csv
    filtered_results = [res for res in results if res["seed_name"] in valid_seed_names]

    if len(filtered_results) < 20:
        print(
            f"Warning: Only found {len(filtered_results)} valid runs for {TASK}. Need at least 20."
        )
        if len(filtered_results) == 0:
            sys.exit(1)

    # Top 10 are the first 10, Bottom 10 are the last 10 in the filtered_results (which is sorted descending)
    top_10 = filtered_results[:10]
    bottom_10 = filtered_results[-10:]
    target_seed_names = set(
        [res["seed_name"] for res in top_10] + [res["seed_name"] for res in bottom_10]
    )

    # Filter the dataframe to only keep those 20 rows
    # Reconstruct the seed name for filtering
    df_full["seed_name"] = [
        f"{TASK}_RES_{row['stratum']}_idx{idx}_PPO_500000steps"
        for idx, row in df_full.iterrows()
    ]
    df = df_full[df_full["seed_name"].isin(target_seed_names)].copy()

    print(
        f"Selected {len(df)} reservoirs (Top {len(top_10)}, Bottom {len(bottom_10)})."
    )

    # ========================================================================
    # STEP A: DYNAMICAL DISTANCES (DSA_state)
    # ========================================================================
    print(f"Loading trajectories for {TASK}...")
    top_seed_names = set([res["seed_name"] for res in top_10])

    Ys = []
    Us = []
    group_labels = []

    for index, row in df.iterrows():
        stratum = row["stratum"]
        seed = int(row["seed"])
        seed_name = row["seed_name"]

        if seed_name in top_seed_names:
            group_labels.append("Top")
        else:
            group_labels.append("Bottom")

        # O_t = U (inputs to reservoir)
        # C_t = X (reservoir states)
        try:
            U = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ot.npy")
            X = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ct.npy")
        except FileNotFoundError:
            print(f"Warning: Missing trajectory for {stratum} seed {seed}, skipping...")
            raise FileNotFoundError(f"Missing trajectory for {stratum} seed {seed}.")

        Ys.append(X)
        Us.append(U)

    print("\nChoosing rank from the data...")
    n_delays = 3
    rank, per_system = choose_rank(Ys, n_delays=n_delays, min_rank=1)
    print(f"  -> using rank={rank} for all systems")

    print(f"Fitting InputDSA on {len(Ys)} networks...")
    dmd_config = dict(n_delays=n_delays, rank=rank, backend="dmdc")

    inputDSA = InputDSA(
        X=Ys,
        X_control=Us,
        dmd_config=dmd_config,
        simdist_config={"compare": "joint", "return_distance_components": True},
    )

    res = inputDSA.fit_score()
    sims_state_joint = res[:, :, 1]

    print("Extracting pairwise DSA_state distances and labels...")
    num_networks = len(Ys)
    dsa_distances = []
    pair_types = []

    for i in range(num_networks):
        for j in range(i + 1, num_networks):
            dsa_distances.append(sims_state_joint[i, j])
            if group_labels[i] == "Top" and group_labels[j] == "Top":
                pair_types.append("Top vs Top")
            elif group_labels[i] == "Bottom" and group_labels[j] == "Bottom":
                pair_types.append("Bottom vs Bottom")
            else:
                pair_types.append("Top vs Bottom")

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

    # Compute pairwise Euclidean distances directly on the 8D Z-scores
    struct_dist_condensed = pdist(Z, metric="euclidean")

    # ========================================================================
    # STEP C: PLOTTING (Scatter + Hexbin)
    # ========================================================================
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # --- Plot A: Scatter Plot with Trendline ---
    # Draw overall trendline first
    sns.regplot(
        x=struct_dist_condensed,
        y=dsa_dist_condensed,
        ax=axes[0],
        scatter=False,
        line_kws={"color": "gray", "linewidth": 1, "linestyle": "--"},
    )

    # Draw points colored by pair type
    palette = {
        "Top vs Top": "#2ecc71",
        "Bottom vs Bottom": "#e74c3c",
        "Top vs Bottom": "#3498db",
    }
    sns.scatterplot(
        x=struct_dist_condensed,
        y=dsa_dist_condensed,
        hue=pair_types,
        palette=palette,
        alpha=0.6,
        s=40,
        ax=axes[0],
    )

    axes[0].set_title(f"{TASK}: Structural vs. DSA_state Distance (Top & Bottom 10)")
    axes[0].set_xlabel("Structural Distance (Normalized 8D Euclidean)")
    axes[0].set_ylabel("DSA_state Distance")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Plot B: 2D Hexbin Density Plot ---
    # With 20 items (190 pairs), hexbin is sparse but still functional
    hb = axes[1].hexbin(
        struct_dist_condensed, dsa_dist_condensed, gridsize=15, cmap="viridis", mincnt=1
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

    axes[1].set_title(f"{TASK} Density: Structural vs. DSA_state (Top & Bottom 10)")
    axes[1].set_xlabel("Structural Distance (Normalized 8D Euclidean)")
    axes[1].set_ylabel("DSA_state Distance")

    plt.tight_layout()
    output_filename = f"inputdsa_top_bottom_10_reservoirs_H1_{TASK}_NoPCA.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Plots successfully saved to {output_filename}")


if __name__ == "__main__":
    main()
