import os
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def parse_all_selected_csv(csv_path="selected.csv"):
    df = pd.read_csv(csv_path)
    # The 'null' motif is parsed as NaN by pandas
    df["stratum"] = df["stratum"].fillna("null")

    configs = []
    for idx, row in df.iterrows():
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
                "csv_idx": idx,
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


def get_exp_name(env, use_res, motif, steps, csv_idx=None):
    if use_res:
        exp_prefix = f"RES_{motif}_idx{csv_idx}_PPO"
    else:
        exp_prefix = "PPO"
    return f"{env}_{exp_prefix}_{steps}steps"


def run_training(env, steps, use_res, motif_cfg=None, force=False):
    if use_res and motif_cfg:
        exp_prefix = f"RES_{motif_cfg['motif']}_idx{motif_cfg['csv_idx']}_PPO"
    else:
        exp_prefix = "PPO"

    exp_name = f"{env}_{exp_prefix}_{steps}steps"

    seed = motif_cfg["seed"] if motif_cfg else 0
    models_dir = os.path.join("rl_only", "models_all", exp_name)
    model_path = os.path.join(models_dir, f"{exp_prefix}_model_seed_{seed}.zip")

    if not force and os.path.exists(model_path):
        print(
            f"Skipping {env} | {exp_prefix} | {steps} steps, model already exists at {model_path}"
        )
        return exp_name, exp_prefix, seed

    cmd = [
        "uv run",
        "train_rl.py",
        "--env_id",
        env,
        "--training_steps",
        str(steps),
        "--use_reservoir",
        str(use_res),
        "--seed",
        str(seed),
        "--log_dir",
        "rl_only/logs_all",
        "--models_dir",
        "rl_only/models_all",
    ]

    if use_res and motif_cfg:
        cmd.extend(
            [
                "--csv_idx",
                str(motif_cfg["csv_idx"]),
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


def extract_tensorboard_scalars(logdir, scalar_name):
    """
    Extracts scalar values from TensorBoard event files in a directory recursively.
    """
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if "events.out.tfevents" in f:
                event_files.append(os.path.join(root, f))
    if not event_files:
        print(f"Warning: No event files found in {logdir}")
        return [], []

    # We take the first event file in the directory
    event_file = event_files[0]

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    if scalar_name not in ea.Tags()["scalars"]:
        print(f"Warning: {scalar_name} not found in tags: {ea.Tags()['scalars']}")
        return [], []

    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    return steps, vals


def plot_results(results, envs, save_path):
    fig, axs = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))

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
        "PPO": "black",
        "assortative": "tab:orange",
        "disassortative": "tab:green",
        "core_periphery": "tab:red",
        "mixed": "tab:purple",
        "null": "tab:brown",
    }

    # Used to track if a motif label has been added to the legend
    legend_added = {motif: False for motif in motif_labels.keys()}

    for idx, env in enumerate(envs):
        ax = axs[idx] if len(envs) > 1 else axs
        ax.set_title(env, fontsize=14)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean episode reward")
        ax.grid(True)

        env_res = results.get(env, [])
        for res in env_res:
            steps = res["steps"]
            vals = res["vals"]
            motif = res["motif"]

            display_label = motif_labels.get(motif, motif)
            color = colors.get(motif, "black")

            # PPO plotted with thicker solid line, ER-MRL with lighter lines
            alpha = 1.0 if motif == "PPO" else 0.3
            linewidth = 3 if motif == "PPO" else 1

            if not legend_added[motif]:
                ax.plot(
                    steps,
                    vals,
                    label=display_label,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
                legend_added[motif] = True
            else:
                ax.plot(steps, vals, color=color, linewidth=linewidth, alpha=alpha)

    # Add shared legend at the bottom
    handles, labels = (axs[0] if len(envs) > 1 else axs).get_legend_handles_labels()

    # We can override the alpha for the legend so it is solid
    import copy

    legend_handles = []
    for h in handles:
        h_copy = copy.copy(h)
        h_copy.set_alpha(1.0)
        h_copy.set_linewidth(3)
        legend_handles.append(h_copy)

    fig.legend(
        legend_handles,
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
        "--dry_run", action="store_true", help="Only do 100 steps for testing"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing models/logs"
    )
    args = parser.parse_args()

    steps = 100 if args.dry_run else args.training_steps
    envs = ["Ant-v4", "HalfCheetah-v4", "Swimmer-v4"]

    # 1. Parse selected.csv
    configs = parse_all_selected_csv("selected.csv")

    # 2. Run Trainings
    # Collect paths for parsing later
    log_dirs = {}

    for env in envs:
        log_dirs[env] = []

        # a. Vanilla PPO
        exp_name, exp_prefix, seed = run_training(
            env, steps, use_res=False, motif_cfg=None, force=args.force
        )
        tb_dir = os.path.join("rl_only", "logs_all", exp_name)
        log_dirs[env].append({"motif": "PPO", "csv_idx": -1, "dir": tb_dir})

        # b. ER-MRL Motifs
        for cfg in configs:
            exp_name, exp_prefix, seed = run_training(
                env, steps, use_res=True, motif_cfg=cfg, force=args.force
            )
            tb_dir = os.path.join("rl_only", "logs_all", exp_name)
            log_dirs[env].append(
                {"motif": cfg["motif"], "csv_idx": cfg["csv_idx"], "dir": tb_dir}
            )

    # 3. Read Tensorboard logs and plot
    all_results = {}
    for env in envs:
        all_results[env] = []
        for l in log_dirs[env]:
            d = l["dir"]
            motif = l["motif"]
            if os.path.exists(d):
                s, v = extract_tensorboard_scalars(d, "rollout/ep_rew_mean")
                if s and v:
                    all_results[env].append(
                        {"motif": motif, "csv_idx": l["csv_idx"], "steps": s, "vals": v}
                    )
            else:
                print(f"Warning: Directory does not exist {d}")

    os.makedirs("results_analysis", exist_ok=True)
    plot_results(all_results, envs, "results_analysis/tasks_reward_plot_all.png")


if __name__ == "__main__":
    main()
