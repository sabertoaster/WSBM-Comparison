import sys

sys.modules["tensorflow"] = None

import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from er_mrl.wrappers import ReservoirWrapper


def parse_all_selected_csv(csv_path="selected.csv"):
    df = pd.read_csv(csv_path)
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


def collect_trajectories(task, configs, dry_run=False):
    os.makedirs("trajectories", exist_ok=True)

    for cfg in configs:
        motif = cfg["motif"]
        csv_idx = cfg["csv_idx"]
        seed = cfg["seed"]

        # Determine model path
        exp_prefix = f"RES_{motif}_idx{csv_idx}_PPO"
        exp_name = f"{task}_{exp_prefix}_500000steps"
        models_dir = os.path.join("rl_only", "models_all", exp_name)
        model_path = os.path.join(models_dir, f"{exp_prefix}_model_seed_{seed}.zip")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        print(f"Loading {task} model for motif {motif}, seed {seed}...")

        # Set up the environment
        def make_env():
            env = gym.make(task)
            env = ReservoirWrapper(
                env,
                seed=seed,
                units=100,
                lr=0.1,
                sr=0.9,
                iss=1.0,
                skip_c=False,
                motif=motif,
                n_communities=cfg["n_communities"],
                hi=cfg["hi"],
                lo=cfg["lo"],
                mid=cfg["mid"],
                sigma=cfg["sigma"],
                connectivity=cfg["connectivity"],
                symmetric=True,
                p_negative=cfg["p_negative"],
            )
            env = Monitor(env)
            return env

        vec_env = DummyVecEnv([make_env])

        # Monkey patch to extract O_t (reservoir input U) and C_t (reservoir state X)
        res_env = vec_env.envs[0].env  # .env to go inside Monitor

        # We'll store trajectories here
        o_t_traj = []
        c_t_traj = []

        class ReservoirTracker:
            def __init__(self, reservoir):
                self.reservoir = reservoir

            def __call__(self, x, *args, **kwargs):
                # x is the input (U), we copy to avoid references being overwritten
                o_t_traj.append(np.copy(np.squeeze(x)))
                out = self.reservoir(x, *args, **kwargs)
                c_t_traj.append(np.copy(out.flatten()))
                return out

            def __getattr__(self, name):
                return getattr(self.reservoir, name)

        res_env.reservoir = ReservoirTracker(res_env.reservoir)

        # Load the model
        model = PPO.load(model_path, env=vec_env)

        obs = vec_env.reset()
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)

        # The tracking call captured the reset step and all environment steps
        # Save to disk
        np.save(f"trajectories/{task}_{motif}_{seed}_Ot.npy", np.array(o_t_traj))
        np.save(f"trajectories/{task}_{motif}_{seed}_Ct.npy", np.array(c_t_traj))

        print(
            f"Saved {len(o_t_traj)} steps to trajectories/{task}_{motif}_{seed}_*.npy"
        )

        # Clean up memory
        del model
        vec_env.close()

        if dry_run:
            print("Dry run complete for this task. Exiting early.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run 1 model per task")
    args = parser.parse_args()

    configs = parse_all_selected_csv("selected.csv")
    tasks = ["Ant-v4", "HalfCheetah-v4", "Swimmer-v4"]

    for task in tasks:
        print(f"=== Starting extraction for {task} ===")
        collect_trajectories(task, configs, args.dry_run)
