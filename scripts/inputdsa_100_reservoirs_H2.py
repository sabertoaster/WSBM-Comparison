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

try:
    from src.utils import choose_rank
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils import choose_rank


def get_stratum_ticks(df):
    """
    Returns the boundaries and centers for the stratum groups to help with plotting.
    """
    strata = df["stratum"].values
    unique_strata = []
    boundaries = [0]
    centers = []

    current_stratum = strata[0]
    unique_strata.append(current_stratum)

    for i, s in enumerate(strata):
        if s != current_stratum:
            boundaries.append(i)
            centers.append((boundaries[-2] + i) / 2.0)
            current_stratum = s
            unique_strata.append(current_stratum)

    boundaries.append(len(strata))
    centers.append((boundaries[-2] + len(strata)) / 2.0)

    return unique_strata, boundaries, centers


def main():
    # Find selected.csv
    if os.path.exists("selected.csv"):
        csv_path = "selected.csv"
    elif os.path.exists("selection/selected.csv"):
        csv_path = "selection/selected.csv"
    else:
        raise FileNotFoundError("selected.csv not found.")

    df = pd.read_csv(csv_path)
    df["stratum"] = df["stratum"].fillna("null")

    # Sort by stratum to stack them together
    df = df.sort_values(by="stratum").reset_index(drop=True)

    unique_strata, boundaries, centers = get_stratum_ticks(df)

    N_DELAYS = 3

    tasks = ["Ant-v4", "HalfCheetah-v4", "Swimmer-v4"]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"PROCESSING TASK: {task}")
        print(f"{'='*60}")

        Ys = []
        Us = []

        # Load trajectories
        missing = 0
        for index, row in df.iterrows():
            stratum = row["stratum"]
            seed = int(row["seed"])

            try:
                U = np.load(f"trajectories/{task}_{stratum}_{seed}_Ot.npy")
                X = np.load(f"trajectories/{task}_{stratum}_{seed}_Ct.npy")
                Ys.append(X)
                Us.append(U)
            except FileNotFoundError:
                print(f"Warning: Missing trajectory for {stratum} seed {seed}.")
                missing += 1

        if missing > 0:
            print(
                f"Skipping {task} due to {missing} missing files. Run collect_obs_context_vec.py first."
            )
            continue

        print("\nChoosing rank from the data...")
        rank, _ = choose_rank(Ys, n_delays=N_DELAYS, min_rank=1)
        print(f"  -> using rank={rank} for all systems")

        dmd_config = dict(n_delays=N_DELAYS, rank=rank, backend="n4sid")

        print(f"Fitting InputDSA for {len(Ys)} networks...")
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

        print("\nPlotting results...")
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
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

            # Draw boundaries between strata to make block structures visible
            for b in boundaries[1:-1]:
                ax[i].axhline(b - 0.5, color="white", linewidth=1, linestyle="--")
                ax[i].axvline(b - 0.5, color="white", linewidth=1, linestyle="--")

            # Place ticks at the center of each stratum block
            ax[i].set_xticks(centers)
            ax[i].set_yticks(centers)
            ax[i].set_xticklabels(unique_strata, rotation=45, ha="right")
            ax[i].set_yticklabels(unique_strata)

        fig.suptitle(
            f"{task}  (n_delays={dmd_config['n_delays']}, rank={dmd_config['rank']})",
            y=1.06,
            fontsize=16,
        )
        plt.tight_layout()

        out_file = f"inputdsa_100_reservoirs_H2_{task}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

        np.savez(
            f"inputdsa_100_reservoirs_H2_{task}.npz",
            labels=df["stratum"].values,
            joint=sims_full,
            state_joint=sims_state_joint,
            control_joint=sims_control_joint,
            state_separate=sims_state_separate,
            control_separate=sims_control_separate,
            rank=rank,
            n_delays=dmd_config["n_delays"],
        )
        print(f"Results saved to {out_file} and inputdsa_100_reservoirs_H2_{task}.npz")


if __name__ == "__main__":
    main()
