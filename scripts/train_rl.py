"""_summary_
python3 rl_only/train_rl.py --env_id HalfCheetah-v4 --training_steps 300000 --use_reservoir True --motif core_periphery --n_communities 3 --hi 0.9 --lo 0.1 --mid 0.5
"""

import os
import sys

# Prevent stable_baselines3 from ever importing tensorboard or tensorflow
# This completely avoids the PyTorch CUDA segmentation fault.
# sys.modules["tensorboard"] = None
sys.modules["tensorflow"] = None

import time
import datetime
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from er_mrl.wrappers import ReservoirWrapper, DeletedVelocityWrapper


def get_random_seed():
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        return int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agents (with optional Reservoir) without Optuna evolution."
    )

    # Environment and training parameters
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Gymnasium Environment ID (e.g., HalfCheetah-v4)",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=300000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate for PPO"
    )
    parser.add_argument(
        "--del_obs",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Use DeletedVelocityWrapper",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel environments to run",
    )

    # Reservoir parameters
    parser.add_argument(
        "--use_reservoir",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Wrap environment with ReservoirWrapper",
    )
    parser.add_argument(
        "--units", type=int, default=100, help="Number of neurons in the Reservoir"
    )
    parser.add_argument(
        "--res_lr", type=float, default=0.1, help="Leak-rate (lr) of the Reservoir"
    )
    parser.add_argument(
        "--res_sr",
        type=float,
        default=0.9,
        help="Spectral radius (sr) of the Reservoir",
    )
    parser.add_argument(
        "--res_iss",
        type=float,
        default=1.0,
        help="Input scaling (iss) of the Reservoir",
    )
    parser.add_argument(
        "--skip_c",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Use skip connection in ReservoirWrapper",
    )
    parser.add_argument(
        "--motif",
        type=str,
        default="assortative",
        help="WSBM motif (assortative, disassortative, core_periphery, mixed)",
    )
    parser.add_argument(
        "--n_communities", type=int, default=4, help="Number of communities in WSBM"
    )
    parser.add_argument("--hi", type=float, default=0.9, help="High block mean")
    parser.add_argument("--lo", type=float, default=0.1, help="Low block mean")
    parser.add_argument(
        "--mid", type=float, default=0.5, help="Mid block mean (for core_periphery)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Within-block weight standard deviation",
    )
    parser.add_argument(
        "--connectivity", type=float, default=0.1, help="Bernoulli edge probability"
    )
    parser.add_argument(
        "--symmetric",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Make reservoir weight matrix symmetric",
    )
    parser.add_argument(
        "--p_negative",
        type=float,
        default=0.0,
        help="Fraction of edges forced negative",
    )

    # Optional explicitly provided seed
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed. If -1, uses SLURM array task ID or 0.",
    )

    parser.add_argument(
        "--csv_idx",
        type=int,
        default=-1,
        help="CSV Index for distinguishing identical motifs",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join("rl_only", "logs"),
        help="Directory to save logs",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=os.path.join("rl_only", "models"),
        help="Directory to save models",
    )

    args = parser.parse_args()

    # Determine seed
    if args.seed != -1:
        seed = args.seed
    else:
        seed = get_random_seed()

    set_random_seed(seed)

    print(f"--- Starting RL Training on {args.env_id} (Seed: {seed}) ---")

    # Directories for results
    if args.use_reservoir == "True":
        exp_prefix = f"RES_{args.motif}"
        if args.csv_idx != -1:
            exp_prefix += f"_idx{args.csv_idx}"
        exp_prefix += "_PPO"
    else:
        exp_prefix = "PPO"

    exp_name = f"{args.env_id}_{exp_prefix}_{args.training_steps}steps"

    logdir = os.path.join(args.log_dir, exp_name)
    models_dir = os.path.join(args.models_dir, exp_name)

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Initialize environment
    skip_c = True if args.skip_c == "True" else False

    def make_env(rank):
        def _init():
            env = gym.make(args.env_id)

            if args.del_obs == "True":
                env = DeletedVelocityWrapper(env)

            if args.use_reservoir == "True":
                # Important: Use the same seed for ReservoirWrapper so all
                # parallel environments share the EXACT SAME reservoir weights
                env = ReservoirWrapper(
                    env,
                    seed=seed,
                    units=args.units,
                    lr=args.res_lr,
                    sr=args.res_sr,
                    iss=args.res_iss,
                    skip_c=skip_c,
                    motif=args.motif,
                    n_communities=args.n_communities,
                    hi=args.hi,
                    lo=args.lo,
                    mid=args.mid,
                    sigma=args.sigma,
                    connectivity=args.connectivity,
                    symmetric=(args.symmetric == "True"),
                    p_negative=args.p_negative,
                )
            env = Monitor(env)
            return env

        return _init

    print(
        f"-> Launching {args.n_envs} parallel environments for better GPU utilization..."
    )
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    if args.del_obs == "True":
        print("-> Using DeletedVelocityWrapper")

    if args.use_reservoir == "True":
        print(
            f"-> Using ReservoirWrapper (units={args.units}, lr={args.res_lr}, sr={args.res_sr}, iss={args.res_iss}, skip_c={skip_c})"
        )
    else:
        print("-> Standard PPO (No Reservoir)")

    # Model Initialization
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        tensorboard_log=logdir,
        seed=seed,
        device="cpu",
    )

    # Training
    start = time.time()
    print(f"Training for {args.training_steps} timesteps...")

    tb_log_name = f"{exp_prefix}_{args.learning_rate}_seed_{seed}"
    model.learn(total_timesteps=args.training_steps, tb_log_name=tb_log_name)

    # Save Model
    model_path = os.path.join(models_dir, f"{exp_prefix}_model_seed_{seed}")
    model.save(model_path)
    print(f"Model saved at {model_path}.zip")

    end = time.time()
    print(f"--- Training completed in {str(datetime.timedelta(seconds=end-start))} ---")
