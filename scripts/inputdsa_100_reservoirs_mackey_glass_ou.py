import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
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

from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes.wsbm_esn import (
    AssortativeESN,
    DisassortativeESN,
    CorePeripheryESN,
    MixedESN,
)


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


def parse_selected_csv(csv_path="selected.csv"):
    df = pd.read_csv(csv_path)
    # The 'null' motif is parsed as NaN by pandas
    df["stratum"] = df["stratum"].fillna("null")

    # Sort by stratum to group identical motifs together
    df = df.sort_values(by="stratum").reset_index(drop=True)

    configs = []
    for _, row in df.iterrows():
        motif = row["stratum"]
        n_communities = int(row["K"])
        connectivity = row["p0"]
        centre_u = row["centre_u"]
        contrast_u = row["contrast_u"]
        sigma_ratio = row["sigma_ratio"]
        p_negative = row["frac_negative"]
        seed = int(row["seed"])
        core_fraction = row.get("core_fraction", 0.2)

        # In wsbm_esn, contrast is typically max_u - min_u
        hi = centre_u + (contrast_u / 2)
        lo = centre_u - (contrast_u / 2)
        mid = centre_u
        sigma = sigma_ratio * centre_u

        configs.append(
            {
                "motif": motif,
                "n_communities": n_communities,
                "hi": hi,
                "lo": lo,
                "mid": mid,
                "sigma": sigma,
                "connectivity": connectivity,
                "p_negative": p_negative,
                "seed": seed,
                "core_fraction": core_fraction,
            }
        )
    return df, configs


N_DELAYS = 10


def ou_noise(n_timesteps, dim=10, theta=0.15, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(dim)
    out = np.empty((n_timesteps, dim))
    for t in range(n_timesteps):
        x = x * (1.0 - theta) + np.sqrt(2.0 * theta) * rng.standard_normal(dim)
        out[t] = x
    return (out - out.mean(axis=0)) / out.std(axis=0)


def choose_rank(Ys, n_delays, energy=0.99, min_rank=3, max_rank=50):
    per_system = []
    for Y in Ys:
        H = np.hstack([Y[i : len(Y) - n_delays + i + 1] for i in range(n_delays)])
        sv = np.linalg.svd(H - H.mean(axis=0), compute_uv=False)
        cum = np.cumsum(sv**2) / np.sum(sv**2)
        per_system.append(int(np.searchsorted(cum, energy)) + 1)
    rank = int(np.clip(max(per_system), min_rank, max_rank))
    return rank, per_system


def build_systems(configs, units=100):
    systems = []
    labels = []

    print("Initializing ESN models...")
    for config in configs:
        motif = config["motif"]

        common_kwargs = {
            "n_communities": config["n_communities"],
            "connectivity": config["connectivity"],
            "sigma": config["sigma"],
            "p_negative": config["p_negative"],
            "seed": config["seed"],
        }

        if motif == "assortative":
            esn = AssortativeESN(
                mu_in=config["hi"], mu_out=config["lo"], units=units, **common_kwargs
            )
        elif motif == "disassortative":
            esn = DisassortativeESN(
                mu_in=config["hi"], mu_out=config["lo"], units=units, **common_kwargs
            )
        elif motif == "core_periphery":
            esn = CorePeripheryESN(
                n_reservoir=units,
                mu_cc=config["hi"],
                mu_cp=config["mid"],
                mu_pp=config["lo"],
                core_fraction=config["core_fraction"],
                **common_kwargs,
            )
        elif motif == "mixed":
            esn = MixedESN(
                hi=config["hi"], lo=config["lo"], units=units, **common_kwargs
            )
        elif motif == "null":
            # Use AssortativeESN with equal means for Erdős-Rényi graph (null model)
            esn = AssortativeESN(
                mu_in=config["mid"], mu_out=config["mid"], units=units, **common_kwargs
            )
        else:
            print(f"Unknown motif: {motif}")
            continue

        systems.append(esn)
        labels.append(motif)

    print(f"Generated {len(systems)} systems.")
    return systems, labels


def main():
    if os.path.exists("selected.csv"):
        csv_path = "selected.csv"
    elif os.path.exists("selection/selected.csv"):
        csv_path = "selection/selected.csv"
    else:
        raise FileNotFoundError("selected.csv not found.")

    df, configs = parse_selected_csv(csv_path)
    unique_strata, boundaries, centers = get_stratum_ticks(df)

    units = 100

    # N4SID requires N_samples > 2 * n_delays * (dim_u + dim_y)
    # If N_DELAYS=10 and dim=10, 2*10*(10+100) = 2200
    # So we set N to 3000 to be perfectly safe from the all-zeros bug.
    N = 3000
    print(f"Generating input timeseries (N={N})...")
    inputs = {
        "mackey_glass": mackey_glass(n_timesteps=N).reshape(-1, 1),
        "ou_noise_10d": ou_noise(N, dim=10, theta=0.15, seed=0),
    }

    for cond_name, U in inputs.items():
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {cond_name}   (input dim = {U.shape[1]})")
        print(f"{'=' * 60}")

        systems, labels = build_systems(configs, units)

        # Run inputs through systems
        Ys = []
        Us = []
        print("Running timeseries through systems...")
        for i, (label, esn) in enumerate(zip(labels, systems)):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(systems)} systems...")
            Y = esn.run(U)
            Ys.append(Y)
            Us.append(U)

        print("\nChoosing rank from the data...")
        rank, per_system = choose_rank(Ys, n_delays=N_DELAYS)
        print(f"  -> using rank={rank} for all systems")
        if rank < 5:
            print(
                "  WARNING: this input excites very few modes. Motif differences"
                " are unlikely to be resolvable in this condition."
            )
        if max(per_system) > rank:
            print(
                f"  NOTE: the data supports rank {max(per_system)}; capped at {rank}"
                " for compute. Raise max_rank in choose_rank() to use more."
            )

        # InputDSA Analysis
        dmd_config = dict(n_delays=N_DELAYS, rank=rank, backend="n4sid")

        print("\nFitting InputDSA...")
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
            f"input: {cond_name}  (dim={U.shape[1]}, n_delays={N_DELAYS}, rank={rank})",
            y=1.06,
            fontsize=16,
        )
        plt.tight_layout()

        out_file = f"inputdsa_100_reservoirs_results_{cond_name}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

        np.savez(
            f"inputdsa_100_reservoirs_results_{cond_name}.npz",
            labels=df["stratum"].values,
            joint=sims_full,
            state_joint=sims_state_joint,
            control_joint=sims_control_joint,
            state_separate=sims_state_separate,
            control_separate=sims_control_separate,
            rank=rank,
            n_delays=N_DELAYS,
            input_dim=U.shape[1],
        )
        print(f"Results saved to {out_file} and .npz")


if __name__ == "__main__":
    main()
