import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def get_models_metadata(results_list, df_full, task, group_name):
    models = []
    for rank_idx, res in enumerate(results_list):
        seed_name = res["seed_name"]

        row = None
        for idx, r in df_full.iterrows():
            if f"{task}_RES_{r['stratum']}_idx{idx}_PPO_500000steps" == seed_name:
                row = r
                row_idx = idx
                break

        if row is not None:
            models.append(
                {
                    "rank": (
                        rank_idx + 1
                        if group_name == "Top"
                        else len(df_full) - len(results_list) + rank_idx + 1
                    ),
                    "stratum": row["stratum"],
                    "seed": int(row["seed"]),
                    "idx": row_idx,
                    "seed_name": seed_name,
                    "performance": res["performance"],
                    "group": group_name,
                }
            )
    return models


TASKS = ["HalfCheetah-v4", "Ant-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run InputDSA on the top and bottom 10 models for each task."
    )
    parser.add_argument(
        "tasks",
        metavar="TASK",
        nargs="*",
        default=TASKS,
        help=f"Tasks to analyze (default: all five tasks). Choices: {', '.join(TASKS)}",
    )
    args = parser.parse_args(argv)
    for task in args.tasks:
        if task not in TASKS:
            parser.error(f"invalid task {task!r}; choose from {', '.join(TASKS)}")

    # 1. Load the 100 selected reservoirs
    if not os.path.exists("selected.csv") and os.path.exists("selection/selected.csv"):
        csv_path = "selection/selected.csv"
    else:
        csv_path = "selected.csv"

    df_full = pd.read_csv(csv_path)
    df_full["stratum"] = df_full["stratum"].fillna("null")

    for task in args.tasks:
        run_task(task, df_full)


def run_task(TASK, df_full):
    # 2. Get performance rankings
    print(f"Calculating performance rankings for {TASK}...")
    results = get_performance(TASK, window_steps=10000, weight_type="linear")

    # Reconstruct valid seed names from the dataframe
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
        sys.exit(1)

    # Top 10 and Bottom 10 models
    top_10 = filtered_results[:10]
    bottom_10 = filtered_results[-10:]

    # Extract metadata to sort by stratum within groups
    top_10_models = get_models_metadata(top_10, df_full, TASK, "Top")
    bottom_10_models = get_models_metadata(bottom_10, df_full, TASK, "Bottom")

    # 3. Sort each group by stratum
    top_10_models.sort(key=lambda x: x["stratum"])
    bottom_10_models.sort(key=lambda x: x["stratum"])

    # Combine them (Top 10 first, then Bottom 10)
    all_models = top_10_models + bottom_10_models

    print("\nSelected Top 10 Models (Sorted by Stratum):")
    for m in top_10_models:
        print(
            f"  [{m['rank']}] {m['stratum']} (idx {m['idx']}, seed {m['seed']}) - Score: {m['performance']:.2f}"
        )

    print("\nSelected Bottom 10 Models (Sorted by Stratum):")
    for m in bottom_10_models:
        print(
            f"  [{m['rank']}] {m['stratum']} (idx {m['idx']}, seed {m['seed']}) - Score: {m['performance']:.2f}"
        )

    # 4. Load trajectories
    Ys = []
    Us = []
    labels = []

    print(f"\nLoading trajectories for {TASK}...")
    for m in all_models:
        stratum = m["stratum"]
        seed = m["seed"]
        idx = m["idx"]
        group = m["group"]

        try:
            # Ot is input (Control)
            U = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ot.npy")
            # Ct is state (State)
            Y = np.load(f"trajectories/{TASK}_{stratum}_{seed}_Ct.npy")
        except FileNotFoundError:
            print(f"Error: Missing trajectory for {stratum} seed {seed}.")
            sys.exit(1)

        Us.append(U)
        Ys.append(Y)
        # Distinguish Top and Bottom in labels
        prefix = "T_" if group == "Top" else "B_"
        labels.append(f"{prefix}{stratum}_{idx}")

    print("\nChoosing rank from the data...")
    n_delays = 3
    rank, per_system = choose_rank(Ys, n_delays=n_delays, min_rank=1)
    print(f"  -> using rank={rank} for all systems")

    print("\nFitting InputDSA...")
    dmd_config = dict(n_delays=n_delays, rank=rank, backend="n4sid")

    inputDSA = InputDSA(
        X=Ys,
        X_control=Us,
        dmd_config=dmd_config,
        simdist_config={"compare": "joint", "return_distance_components": True},
    )

    res = inputDSA.fit_score()
    sims_full = res[:, :, 0]
    sims_state_joint = res[:, :, 1]
    sims_control_joint = res[:, :, 2]

    inputDSA.update_compare_method(compare="state")
    sims_state_separate = inputDSA.score()

    inputDSA.update_compare_method(
        compare="control", simdist_config={"score_method": "euclidean"}
    )
    sims_control_separate = inputDSA.score()

    # 6. Plot and Save
    print("\nPlotting results...")
    fig, ax = plt.subplots(1, 5, figsize=(30, 6))
    sims_data = [
        sims_full,
        sims_state_joint,
        sims_control_joint,
        sims_state_separate,
        sims_control_separate,
    ]
    titles = [
        "Joint",
        "State (Joint)",
        "Control (Joint)",
        "State (Separate)",
        "Control (Separate)",
    ]

    for i, (data, title) in enumerate(zip(sims_data, titles)):
        im = ax[i].imshow(data, cmap="viridis")
        cbar = plt.colorbar(im, ax=ax[i], shrink=0.7, location="top")
        ax[i].set_title(title, y=1.2, pad=10)
        ax[i].set_xticks(range(len(labels)))
        ax[i].set_yticks(range(len(labels)))
        ax[i].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax[i].set_yticklabels(labels, fontsize=8)

        # Add visual separation between Top 10 and Bottom 10
        ax[i].axhline(y=9.5, color="red", linestyle="--", linewidth=2, alpha=0.8)
        ax[i].axvline(x=9.5, color="red", linestyle="--", linewidth=2, alpha=0.8)

    fig.suptitle(
        f"InputDSA on Top 10 & Bottom 10 {TASK} Models (Sorted by Stratum) | rank={rank}, n_delays={n_delays}",
        y=1.08,
    )
    plt.tight_layout()

    out_file = f"inputdsa_top_bottom_10_{TASK}_sorted.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        f"inputdsa_top_bottom_10_{TASK}_sorted.npz",
        labels=np.array(labels),
        joint=sims_full,
        state_joint=sims_state_joint,
        control_joint=sims_control_joint,
        state_separate=sims_state_separate,
        control_separate=sims_control_separate,
        rank=rank,
        n_delays=n_delays,
    )
    print(f"Results saved to {out_file} and .npz")


if __name__ == "__main__":
    main()
