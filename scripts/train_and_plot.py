import os
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_selected_csv(csv_path="selected.csv"):
    df = pd.read_csv(csv_path)
    # The 'null' motif is parsed as NaN by pandas
    df["stratum"] = df["stratum"].fillna("null")

    # Get first instance of each stratum
    first_instances = df.groupby("stratum").first().reset_index()

    configs = []
    for _, row in first_instances.iterrows():
        motif = row["stratum"]
        n_communities = int(row["K"])
        connectivity = row["p0"]
        centre_u = row["centre_u"]
        contrast_u = row["contrast_u"]
        sigma_ratio = row["sigma_ratio"]
        p_negative = row["frac_negative"]
        seed = int(row["seed"])

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
            }
        )
    return configs


def get_exp_name(env, use_res, motif, steps):
    exp_prefix = f"RES_{motif}_PPO" if use_res else "PPO"
    return f"{env}_{exp_prefix}_{steps}steps"


def run_training(env, steps, use_res, seed, motif_cfg=None, force=False):
    if use_res and motif_cfg:
        exp_prefix = f"RES_{motif_cfg['motif']}_PPO"
    else:
        exp_prefix = "PPO"
    
    exp_name = f"{env}_{exp_prefix}_{steps}steps"

    models_dir = os.path.join("rl_only", "models", exp_name)
    model_path = os.path.join(models_dir, f"{exp_prefix}_model_seed_{seed}.zip")

    if not force and os.path.exists(model_path):
        print(
            f"Skipping {env} | {exp_prefix} | {steps} steps, model already exists at {model_path}"
        )
        return exp_name, exp_prefix, seed

    cmd = [
        "uv run",
        "scripts/train_rl.py",
        "--env_id",
        env,
        "--training_steps",
        str(steps),
        "--use_reservoir",
        str(use_res),
        "--seed",
        str(seed),
    ]

    if use_res and motif_cfg:
        cmd.extend(
            [
                "--motif",
                motif_cfg["motif"],
                "--n_communities",
                str(motif_cfg["n_communities"]),
                "--hi",
                str(motif_cfg["hi"]),
                "--lo",
                str(motif_cfg["lo"]),
                "--mid",
                str(motif_cfg["mid"]),
                "--sigma",
                str(motif_cfg["sigma"]),
                "--connectivity",
                str(motif_cfg["connectivity"]),
                "--p_negative",
                str(motif_cfg["p_negative"]),
            ]
        )

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return exp_name, exp_prefix, seed


def extract_tensorboard_scalars_all_seeds(logdir, scalar_name):
    """
    Extracts scalar values from TensorBoard event files in a directory recursively.
    Returns common steps, mean values, and standard deviation across seeds.
    """
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if "events.out.tfevents" in f:
                event_files.append(os.path.join(root, f))
    if not event_files:
        print(f"Warning: No event files found in {logdir}")
        return [], [], []

    all_vals = []
    common_steps = None

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        if scalar_name not in ea.Tags()["scalars"]:
            continue

        events = ea.Scalars(scalar_name)
        steps = [e.step for e in events]
        vals = [e.value for e in events]

        all_vals.append(vals)
        if common_steps is None:
            common_steps = steps

    if not all_vals:
        return [], [], []

    # Truncate to the minimum length in case some runs ended slightly early
    min_len = min(len(v) for v in all_vals)
    common_steps = common_steps[:min_len]
    truncated_vals = [v[:min_len] for v in all_vals]

    arr = np.array(truncated_vals)
    mean_vals = np.mean(arr, axis=0)
    std_vals = np.std(arr, axis=0)

    return common_steps, mean_vals, std_vals


def plot_results(results, envs, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Prettier names for legend
    motif_labels = {
        "PPO": "Vanilla RL",
        "assortative": "AssortativeESN",
        "disassortative": "DisassortativeESN",
        "core_periphery": "Core-peripheryESN",
        "mixed": "MixedESN",
        "null": "RandomESN",
    }

    colors = {
        "PPO": "tab:blue",
        "assortative": "tab:orange",
        "disassortative": "tab:green",
        "core_periphery": "tab:red",
        "mixed": "tab:purple",
        "null": "tab:brown",
    }

    for idx, env in enumerate(envs):
        ax = axs[idx]
        ax.set_title(env, fontsize=14)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean episode reward")
        ax.grid(True)

        env_res = results.get(env, [])
        for res in env_res:
            steps = res["steps"]
            mean_vals = res["mean_vals"]
            std_vals = res["std_vals"]
            label = res["label"]

            display_label = motif_labels.get(label, label)
            color = colors.get(label, "black")

            # Plot mean as solid line and standard deviation as shaded region
            ax.plot(steps, mean_vals, label=display_label, color=color, linewidth=2)
            ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)

    # Add shared legend at the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_steps", type=int, default=500000)
    parser.add_argument(
        "--dry_run", action="store_true", help="Only do 1000 steps for testing"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing models/logs"
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds to run per configuration"
    )
    args = parser.parse_args()

    steps = 1000 if args.dry_run else args.training_steps
    envs = ["Ant-v4", "HalfCheetah-v4", "Swimmer-v4"]

    # 1. Parse selected.csv
    configs = parse_selected_csv("selected.csv")

    # 2. Run Trainings
    # Collect paths for parsing later
    log_dirs = {}

    for env in envs:
        log_dirs[env] = []

        # a. Vanilla PPO
        exp_name = None
        for i in range(args.seeds):
            exp_name, exp_prefix, seed = run_training(
                env, steps, use_res=False, seed=i, motif_cfg=None, force=args.force
            )
        if exp_name:
            tb_dir = os.path.join("rl_only", "logs", exp_name)
            log_dirs[env].append({"label": "PPO", "dir": tb_dir})

        # b. Motifs
        for cfg in configs:
            for i in range(args.seeds):
                # Incorporate 'i' to ensure distinct seeds
                seed = cfg["seed"] + i
                exp_name, exp_prefix, seed = run_training(
                    env, steps, use_res=True, seed=seed, motif_cfg=cfg, force=args.force
                )
            if exp_name:
                tb_dir = os.path.join("rl_only", "logs", exp_name)
                log_dirs[env].append({"label": cfg["motif"], "dir": tb_dir})

    # 3. Read Tensorboard logs and plot
    all_results = {}
    for env in envs:
        all_results[env] = []
        for l in log_dirs[env]:
            d = l["dir"]
            label = l["label"]
            if os.path.exists(d):
                s, mean_v, std_v = extract_tensorboard_scalars_all_seeds(d, "rollout/ep_rew_mean")
                if len(s) > 0:
                    all_results[env].append({"label": label, "steps": s, "mean_vals": mean_v, "std_vals": std_v})
            else:
                print(f"Warning: Directory does not exist {d}")

    os.makedirs("results_analysis", exist_ok=True)
    plot_results(all_results, envs, "results_analysis/tasks_reward_plot_top5.png")


if __name__ == "__main__":
    main()
