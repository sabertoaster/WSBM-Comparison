import os
import glob
import numpy as np
import warnings

# Suppress annoying tensorboard warnings if possible
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
except ImportError:
    pass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_DIR = "rl_only/logs_all"
MODELS_DIR = "rl_only/models_all"

tasks = ["Swimmer-v4", "Hopper-v4", "Walker2d-v4"]


def compute_weighted_mean(scalars, window_steps=10000, weight_type="linear"):
    """
    Computes a weighted mean of the scalars in the last `window_steps`.
    """
    if not scalars:
        return float("-inf")

    max_step = max(s.step for s in scalars)
    min_step_window = max_step - window_steps

    window_scalars = [s for s in scalars if s.step >= min_step_window]

    if not window_scalars:
        return scalars[-1].value

    values = np.array([s.value for s in window_scalars])
    steps = np.array([s.step for s in window_scalars])

    if weight_type == "linear":
        if len(window_scalars) == 1:
            weights = np.array([1.0])
        else:
            # Linear weight from near 0 to 1 at max_step
            weights = (steps - min_step_window) / window_steps
            weights = np.clip(weights, 1e-6, 1.0)
    elif weight_type == "exponential":
        # Exponential weight
        weights = np.exp((steps - max_step) / (window_steps / 3.0))
    elif weight_type == "mean":
        weights = np.ones_like(values)
    else:
        weights = np.ones_like(values)

    weights /= weights.sum()
    return np.sum(values * weights)


def get_performance(task_prefix, window_steps=10000, weight_type="linear"):
    search_path = os.path.join(LOGS_DIR, f"{task_prefix}_*")
    log_dirs = glob.glob(search_path)

    # If the task itself matches (e.g. Ant-v4_PPO_500000steps), we also want it.
    # The glob with wildcard captures everything starting with Ant-v4_

    results = []

    for log_dir in log_dirs:
        seed_name = os.path.basename(log_dir)

        inner_dirs = [
            d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))
        ]
        if not inner_dirs:
            events = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            if not events:
                continue
            event_file = events[0]
        else:
            inner_dir = inner_dirs[0]
            events = glob.glob(
                os.path.join(log_dir, inner_dir, "events.out.tfevents.*")
            )
            if not events:
                continue
            event_file = events[0]

        # Initialize EventAccumulator to parse the file
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()

        try:
            scalars = ea.Scalars("rollout/ep_rew_mean")
            perf = compute_weighted_mean(
                scalars, window_steps=window_steps, weight_type=weight_type
            )
        except KeyError:
            # This scalar might not exist if the run failed or is empty
            perf = float("-inf")

        model_search = os.path.join(MODELS_DIR, seed_name, "*.zip")
        models = glob.glob(model_search)
        model_path = models[0] if models else "Model not found"

        results.append(
            {
                "seed_name": seed_name,
                "performance": perf,
                "log_path": log_dir,
                "model_path": model_path,
            }
        )

    # Sort descending by performance
    results.sort(key=lambda x: x["performance"], reverse=True)
    return results


if __name__ == "__main__":
    # Configuration
    WEIGHT_TYPE = "linear"  # can be 'linear', 'exponential', or 'mean'
    WINDOW_STEPS = 10000

    print(f"Using '{WEIGHT_TYPE}' weighted mask over the last {WINDOW_STEPS} steps.")

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print(f"{'='*70}")

        results = get_performance(
            task, window_steps=WINDOW_STEPS, weight_type=WEIGHT_TYPE
        )

        if not results:
            print("No results found.")
            continue

        print("\n" + "*" * 20 + " TOP 10 " + "*" * 20)
        for i, res in enumerate(results[:10]):
            print(f"[{i+1}] {res['seed_name']}")
            print(f"    Score: {res['performance']:.2f}")
            print(f"    Log:   {res['log_path']}")
            print(f"    Model: {res['model_path']}")

        print("\n" + "*" * 20 + " BOTTOM 10 " + "*" * 20)
        # Sort the bottom 10 so the absolute worst is listed first
        bottom_10 = sorted(results[-10:], key=lambda x: x["performance"])
        total_runs = len(results)

        for i, res in enumerate(bottom_10):
            rank = total_runs - i
            print(f"[Worst {i+1} | Rank {rank}] {res['seed_name']}")
            print(f"    Score: {res['performance']:.2f}")
            print(f"    Log:   {res['log_path']}")
            print(f"    Model: {res['model_path']}")
