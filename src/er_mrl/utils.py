"""Legacy ER-MRL plotting and compatibility helpers.

New experiments use src.utils for logs and seeds. Interactive plotting saves to
new artifact directories; deletion remains an explicitly confirmed manual call.
"""

import shutil

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


def get_logs_values(logdir, models, nb_seeds):
    """Read legacy per-seed TensorBoard runs and align reward curves by step.

    Returns dictionaries of mean, standard deviation, and recorded step arrays.
    Each seed is loaded independently, so restarts are not counted as replicates.
    """
    from pathlib import Path

    from src.utils import align_curves, read_scalars

    means, deviations, timesteps = {}, {}, {}
    for model in models:
        curves = []
        for seed in range(nb_seeds):
            directories = sorted(
                p
                for p in Path(logdir).iterdir()
                if p.is_dir() and p.name.startswith(model + "_") and p.name.endswith(f"{seed}_1")
            )
            events = [read_scalars(directory) for directory in directories]
            for steps, values in events:
                if len(steps):
                    curves.append((steps, values))
        steps, mean, std = align_curves(curves)
        means[model], deviations[model], timesteps[model] = mean, std, steps
    return means, deviations, timesteps


def plot_results(exp_name, logdir, models, nb_seeds, max_reward=None):
    """Plot legacy per-model reward means and deviations from TensorBoard events."""
    models_mean_arrays, models_std_arrays, timesteps = get_logs_values(logdir, models, nb_seeds)

    plt.figure(figsize=(15, 6))
    plt.title(f"Evolution of mean episode reward on '{exp_name}'")
    plt.xlabel("Steps")
    plt.ylabel("Mean episode reward")

    for model in models:
        plt.plot(timesteps[model], models_mean_arrays[model], label=model)
        plt.fill_between(
            timesteps[model],
            models_mean_arrays[model] - models_std_arrays[model],
            models_mean_arrays[model] + models_std_arrays[model],
            alpha=0.2,
        )

    if max_reward:
        plt.plot(
            timesteps[model],
            np.full(timesteps[model].shape[0], max_reward),
            color="black",
            linestyle="dashed",
            alpha=0.5,
            label="Max reward",
        )

    plt.legend()
    plt.show()


def delete_logs_models(exp_name):
    """Ask for interactive confirmation before deleting manually specified legacy logs and models."""
    logdir = f"logs/{exp_name}/"
    models_dir = f"models/{exp_name}/"

    confirmation = input(f"Are you sure you want to delete the directories for {exp_name} ? (Y/N)")
    if confirmation == "Y":
        shutil.rmtree(logdir)
        shutil.rmtree(models_dir)
        print("directories deleted")
    else:
        pass


def random_agent_avg_reward(env_id, n_episodes):
    """Return mean episodic total reward for a random-action policy."""
    env = gym.make(env_id)
    mean_ep_rewards = []

    for ep in range(n_episodes):
        env.reset()
        ep_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        mean_ep_rewards.append(ep_reward)

    return np.mean(mean_ep_rewards)


def plot_episode_obs(
    dones,
    obs_history,
    obs_idx_names,
    context_history,
    episode_id,
    ep_timesteps=None,
    legend=True,
    figsize=(8, 4),
    output_dir=None,
    n_context=20,
):
    """Plot observation features and up to n_context reservoir neurons for one episode; save fresh figures."""
    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0

    ep_step_stop = dones[episode_id]
    if ep_timesteps:
        nb_steps = ep_timesteps
    else:
        nb_steps = ep_step_stop - ep_step_start

    timesteps = np.arange(stop=nb_steps)

    """Plot the RL agents Observation """

    plt.figure(figsize=figsize)
    # Remove comment below if also wanna plot the actions
    # plt.plot(timesteps, actions_history[ep_step_start:ep_step_stop], label=action, alpha=0.5)

    for idx in range(len(obs_idx_names)):
        if ep_timesteps:
            observation = obs_history[ep_step_start : ep_step_start + ep_timesteps, idx]
        else:
            observation = obs_history[ep_step_start:ep_step_stop, idx]
        plt.plot(timesteps, observation, label=obs_idx_names[idx])
        plt.title(f"Observations o_t Episode {episode_id}")
        if legend:
            plt.legend()

    plt.savefig(_figure_path(output_dir, f"obs_ep_{episode_id}"))
    plt.show()

    """Plot the ER-MRL agents Context"""

    plt.figure(figsize=figsize)
    # Remove comment below if also wanna plot the actions
    # plt.plot(timesteps, actions_history[ep_step_start:ep_step_stop], label=action, alpha=0.5)

    obs_RES_neurons = min(n_context, context_history.shape[1])
    for idx in range(obs_RES_neurons):
        if ep_timesteps:
            context = context_history[ep_step_start : ep_step_start + ep_timesteps, idx]
        else:
            context = context_history[ep_step_start:ep_step_stop, idx]
        plt.plot(timesteps, context)
        plt.title(f"Contexts c_t Episode {episode_id}")

    plt.savefig(_figure_path(output_dir, f"ctx_ep_{episode_id}"))
    plt.show()

    ep_step_start += ep_step_stop - ep_step_start


def plot_episode_fft(dones, obs_history, context_history, episode_id, n_context=100):
    """Plot FFT magnitudes of mean observation and context signals for one episode."""
    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0

    ep_step_stop = dones[episode_id]
    """Plot the fft of RL agents observations"""

    observation = obs_history[ep_step_start:ep_step_stop, :]
    sum_observations = np.sum(observation, axis=1)

    # Division factor for intensity
    sum_observations = sum_observations / obs_history.shape[1]

    fourier = fft(sum_observations)
    # Plot the result (the spectrum |Xk|)
    plt.figure(figsize=(9, 3))
    plt.ylim(0, 10)
    plt.plot(np.abs(fourier))
    plt.title(f"FFT Sum_observations Episode {episode_id}")
    plt.show()

    """Plot the fft of ER-MRL agents context"""

    obs_RES_neurons = min(n_context, context_history.shape[1])
    context = context_history[ep_step_start:ep_step_stop, :obs_RES_neurons]

    sum_contexts = np.sum(context, axis=1)

    sum_contexts = sum_contexts / context_history.shape[1]

    fourier = fft(sum_contexts)
    # Plot the result (the spectrum |Xk|)
    plt.figure(figsize=(9, 3))
    plt.ylim(0, 10)
    plt.plot(np.abs(fourier))
    plt.title(f"FFT Sum_Context Episode {episode_id}")
    plt.show()

    ep_step_start += ep_step_stop - ep_step_start


def plot_observations_context(
    dones,
    obs_history,
    obs_idx_names,
    context_history,
    episode_id,
    ep_timesteps=None,
    legend=True,
    figsize=(16, 8),
    output_dir=None,
    n_context=20,
):
    """Plot observations and up to n_context reservoir neurons together; save a fresh figure."""
    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0
    ep_step_stop = dones[episode_id]

    if ep_timesteps:
        nb_steps = ep_timesteps
    else:
        nb_steps = ep_step_stop - ep_step_start

    timesteps = np.arange(stop=nb_steps)

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    """ Observation """
    ax_obs = axes[0]
    ax_obs.set_title(f"Observations Episode {episode_id}")

    for idx in range(len(obs_idx_names)):
        if ep_timesteps:
            observation = obs_history[ep_step_start : ep_step_start + ep_timesteps, idx]
        else:
            observation = obs_history[ep_step_start:ep_step_stop, idx]
        ax_obs.plot(timesteps, observation, label=obs_idx_names[idx])
        if legend:
            ax_obs.legend()

    """ RES_Context"""
    ax_ctx = axes[1]
    ax_ctx.set_title(f"RES_Context Episode {episode_id}")

    obs_RES_neurons = min(n_context, context_history.shape[1])
    for idx in range(obs_RES_neurons):
        if ep_timesteps:
            context = context_history[ep_step_start : ep_step_start + ep_timesteps, idx]
        else:
            context = context_history[ep_step_start:ep_step_stop, idx]
        ax_ctx.plot(timesteps, context)

    plt.tight_layout()
    plt.savefig(_figure_path(output_dir, f"combined_plots_ep_{episode_id}"))
    plt.show()
    plt.close()  # Close the figure to avoid displaying it immediately

    ep_step_start += ep_step_stop - ep_step_start


def get_random_seed():
    """Return the SLURM array task seed, or zero outside an array job."""
    from src.utils import get_random_seed as shared_seed

    return shared_seed()


def get_models(del_obs):
    """Return historical model labels and learning rates for the requested masking preset."""
    if del_obs == "True":
        lrs = [0.0003, 0.0003, 0.0001, 0.00005]
        models = ["PPO", "RES_PPO_0.0003", "RES_PPO_0.0001", "RES_PPO_0.00005"]
    elif del_obs == "False":
        lrs = [0.0003, 0.0003, 0.0001]
        models = ["PPO", "RES_PPO_0.0003", "RES_PPO_0.0001"]
    else:
        raise (ValueError("Unknown type of envs"))

    return lrs, models


def _figure_path(output_dir, name):
    """Allocate a fresh figure path, preserving any existing output."""
    from pathlib import Path
    from uuid import uuid4

    from src.utils import ROOT

    directory = (
        Path(output_dir)
        if output_dir is not None
        else ROOT / "artifacts" / "legacy" / f"interactive-{uuid4().hex[:8]}" / "figures"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    if path.exists():
        path = directory / f"{name}-{uuid4().hex[:8]}.png"
    return path
