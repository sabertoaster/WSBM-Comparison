"""Seeded deterministic evaluation and episode-separated trajectory collection.

Corrected records contain the state before each transition and the input used
for that transition. Terminal states are saved separately to retain alignment.
"""

import numpy as np
import pandas as pd

from src.training import make_environment, registry_environment
from src.utils import configure_tensorboard, file_hash, matrix_hash


def rollout(model, env, seed, collect=False):
    """Return one episode's total reward, status, and optional (state,input) arrays.

    Uses a scalar Gymnasium environment, avoiding vector-environment autoresets.
    """
    observation, _ = env.reset(seed=seed)
    states, inputs, post_states = [], [], []
    wrapper = env.env
    total, length = 0.0, 0
    while True:
        previous = wrapper.last_context.copy() if collect else None
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        if collect:
            states.append(previous)
            inputs.append(wrapper.last_input.copy())
            post_states.append(wrapper.last_context.copy())
        total += float(reward)
        length += 1
        if terminated or truncated:
            break
    return {
        "return": total,
        "length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "states": np.asarray(states),
        "inputs": np.asarray(inputs),
        "post_states": np.asarray(post_states),
        "terminal_state": wrapper.last_context.copy() if collect else np.array([]),
    }


def evaluate_registry(args, frame, registry, output, collect=False):
    """Evaluate each replicate on shared episode seeds and optionally save trajectories."""
    configure_tensorboard()
    from stable_baselines3 import PPO

    records, trajectories = [], []
    for _, model_record in registry.iterrows():
        config, row = registry_environment(model_record, args, frame)
        baseline = model_record.stratum == "PPO"
        env = make_environment(model_record.task, config, row, args.selected_csv, baseline, args.evaluation_seed)
        try:
            model = PPO.load(model_record.model_path, env=env, device=args.device)
            if not baseline:
                realized_hash = matrix_hash(env.env.reservoir.W)
                if model_record.get("matrix_hash", "") and realized_hash != model_record.matrix_hash:
                    raise ValueError("Evaluation reservoir does not match training matrix")
                if (
                    model_record.get("readin_hash", "")
                    and matrix_hash(env.env.reservoir.Win) != model_record.readin_hash
                ):
                    raise ValueError("Evaluation read-in weights do not match training")
            for episode in range(args.evaluation_episodes):
                seed = args.evaluation_seed + episode
                result = rollout(model, env, seed, collect and not baseline)
                common = {
                    "task": model_record.task,
                    "csv_idx": int(model_record.csv_idx),
                    "stratum": model_record.stratum,
                    "policy_seed": int(model_record.policy_seed),
                    "reservoir_seed": int(model_record.reservoir_seed),
                    "episode": episode,
                    "evaluation_seed": seed,
                    "protocol": args.protocol,
                    "selection_sha256": file_hash(args.selected_csv),
                }
                records.append(
                    {**common, **{key: result[key] for key in ("return", "length", "terminated", "truncated")}}
                )
                if collect and not baseline:
                    identifier = f"{model_record.task}_idx{model_record.csv_idx}_policy{model_record.policy_seed}_episode{episode}"
                    target = output / "trajectories" / f"{identifier}.npz"
                    if args.protocol == "corrected":
                        states = np.vstack([result["states"], result["terminal_state"]])
                        inputs = np.vstack([result["inputs"], result["inputs"][-1]])
                    else:
                        # Legacy InputDSA paired input and post-input context at the same index.
                        states, inputs = result["post_states"], result["inputs"]
                    np.savez_compressed(
                        target,
                        states=states,
                        inputs=inputs,
                        protocol=args.protocol,
                        **{key: value for key, value in common.items() if key != "protocol"},
                    )
                    trajectories.append({**common, "path": str(target.resolve()), "matrix_hash": realized_hash})
                pd.DataFrame(records).to_csv(output / "evaluation" / "episodes.csv", index=False)
        finally:
            env.close()
    if trajectories:
        pd.DataFrame(trajectories).to_csv(output / "trajectories" / "trajectories.csv", index=False)
    result_frame = pd.DataFrame(records)
    summary = (
        result_frame.groupby(["task", "csv_idx", "stratum", "policy_seed"], as_index=False)["return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(output / "evaluation" / "returns.csv", index=False)
    return result_frame


def random_scores(args, output):
    """Evaluate seeded random policies; legacy reports mean per-step rewards too."""
    import gymnasium as gym

    records = []
    for task in args.tasks:
        env = gym.make(task, max_episode_steps=args.max_episode_steps) if args.max_episode_steps else gym.make(task)
        try:
            for episode in range(args.evaluation_episodes):
                seed = args.evaluation_seed + episode
                env.reset(seed=seed)
                env.action_space.seed(seed)
                total, length = 0.0, 0
                while True:
                    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                records.append(
                    {
                        "task": task,
                        "episode": episode,
                        "seed": seed,
                        "return": total,
                        "mean_step_reward": total / length,
                        "length": length,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
        finally:
            env.close()
    frame = pd.DataFrame(records)
    frame.to_csv(output / "evaluation" / "random_scores.csv", index=False)
    return frame
