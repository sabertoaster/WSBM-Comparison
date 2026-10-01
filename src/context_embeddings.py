"""Agent-ranked, episode-aligned reservoir activity embeddings.

Contexts are sampled after each transition: c[t+1] is paired with the masked
observation o[t+1], executed action a[t], and reward r[t]. Reset samples are
excluded. Behavior-only CEBRA uses a one-sample encoder and delta sampling so
concatenating episodes does not introduce artificial temporal neighbors.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.evaluation import encoder_fingerprint, evaluate_registry
from src.figures import pyplot
from src.training import load_registry, make_environment, registry_environment
from src.utils import file_hash, matrix_hash, write_json

AGENT_KEYS = ["task", "csv_idx", "policy_seed"]
EPISODE_KEYS = [*AGENT_KEYS, "evaluation_seed"]
SIGNALS = ("observations", "actions", "rewards")


def validate_sources(args, frame):
    """Validate registries and explicit caches before creating an output run."""
    registry = load_registry(args, frame)
    registry = registry[registry.csv_idx >= 0].copy()
    if registry.empty or registry.duplicated(AGENT_KEYS).any():
        raise ValueError("Need a nonempty registry with unique task/reservoir/policy agents")
    registry["encoder_sha256"] = [
        encoder_fingerprint(registry_environment(record, args, frame)[0]) for _, record in registry.iterrows()
    ]
    for task in args.tasks:
        count = int(registry.task.eq(task).sum())
        required = args.k * (2 if args.group == "both" else 1)
        if count < required:
            raise ValueError(f"{task}: need {required} agents, found {count}")
    for key in ("evaluation_csv", "trajectories_csv"):
        path = getattr(args, key)
        if path:
            validate_cache_manifest(path, args)
    if args.evaluation_csv:
        read_evaluation(args.evaluation_csv, args, registry)
    if args.trajectories_csv:
        read_trajectory_registry(args.trajectories_csv, args)
    return registry


def validate_cache_manifest(path, args):
    """Require the originating manifest and the same selection/protocol."""
    manifest = Path(path).parent.parent / "manifest.json"
    metadata = json.loads(manifest.read_text())
    if metadata.get("selection_sha256") != file_hash(args.selected_csv):
        raise ValueError("Cached selection fingerprint mismatch")
    if metadata.get("arguments", {}).get("protocol") != args.protocol:
        raise ValueError("Cached protocol mismatch")
    if metadata.get("status") != "completed":
        raise ValueError("Cached run must be completed")
    saved = metadata["arguments"]
    # A legacy reservoir carries state between episodes; changing the seed list
    # changes later episodes even when an individual seed matches.
    for key in ("evaluation_seed", "evaluation_episodes"):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"Cached {key} mismatch; use the originating evaluation settings")
    if args.protocol == "legacy":
        for key in ("units", "res_lr", "res_sr", "res_iss", "del_obs", "skip_c", "reset_res", "max_episode_steps"):
            if key in saved and saved[key] != getattr(args, key):
                raise ValueError(f"Cached legacy {key} mismatch")
    if args.max_episode_steps is not None and saved.get("max_episode_steps") != args.max_episode_steps:
        raise ValueError("Cached episode time limit mismatch")
    return metadata


def read_evaluation(path, args, registry):
    """Load compatible episode returns, permitting absent agents to be evaluated."""
    episodes = pd.read_csv(path, keep_default_na=False)
    required = {*EPISODE_KEYS, "return", "protocol", "reservoir_seed", "selection_sha256"}
    if required - set(episodes) or not episodes.protocol.eq(args.protocol).all():
        raise ValueError("Evaluation columns or protocol mismatch")
    if not episodes.selection_sha256.eq(file_hash(args.selected_csv)).all():
        raise ValueError("Evaluation selection fingerprint mismatch")
    episodes = episodes.merge(registry[AGENT_KEYS], on=AGENT_KEYS, validate="many_to_one")
    if episodes.duplicated(EPISODE_KEYS).any() or not np.isfinite(episodes["return"].astype(float)).all():
        raise ValueError("Duplicate evaluation episodes or nonfinite returns")
    expected_seeds = set(range(args.evaluation_seed, args.evaluation_seed + args.evaluation_episodes))
    for _, record in registry.iterrows():
        subset = agent_rows(episodes, record)
        if subset.empty:
            continue
        if set(subset.evaluation_seed) != expected_seeds:
            raise ValueError("Cached agent lacks the requested common evaluation episodes")
        if not subset.reservoir_seed.eq(record.reservoir_seed).all():
            raise ValueError("Evaluation reservoir seed mismatch")
        for key, actual in (
            ("model_sha256", lambda: file_hash(record.model_path)),
            ("encoder_sha256", lambda: record.encoder_sha256),
        ):
            if key in subset:
                hashes = subset.loc[subset[key].ne(""), key]
                if len(hashes) and not hashes.eq(actual()).all():
                    raise ValueError(
                        f"Evaluation {'checkpoint' if key == 'model_sha256' else 'encoder'} fingerprint mismatch"
                    )
    return episodes


def read_trajectory_registry(path, args):
    """Read episode metadata without guessing identity from filenames."""
    registry = pd.read_csv(path, keep_default_na=False)
    required = {*EPISODE_KEYS, "path", "protocol", "selection_sha256", "reservoir_seed"}
    if required - set(registry) or not registry.protocol.eq(args.protocol).all():
        raise ValueError("Trajectory columns or protocol mismatch")
    if not registry.selection_sha256.eq(file_hash(args.selected_csv)).all():
        raise ValueError("Trajectory selection fingerprint mismatch")
    if registry.duplicated(EPISODE_KEYS).any():
        raise ValueError("Duplicate trajectory episodes")
    return registry


def agent_rows(table, record):
    """Match an individual policy without averaging across optimization seeds."""
    return table.loc[np.logical_and.reduce([table[key].eq(record[key]) for key in AGENT_KEYS])]


def select_agents(episodes, k=1, group="top"):
    """Rank individual agents by episode mean with stable identity tie breaks."""
    if episodes.empty or not np.isfinite(episodes["return"].to_numpy(dtype=float)).all():
        raise ValueError("Need finite evaluation returns")
    scores = episodes.groupby(AGENT_KEYS, as_index=False)["return"].agg(
        mean_return="mean", return_std="std", episode_count="count"
    )
    selections, rankings = [], []
    for task, values in scores.groupby("task", sort=True):
        ordered = values.sort_values(["mean_return", "csv_idx", "policy_seed"], ascending=[False, True, True])
        ordered = ordered.assign(rank=np.arange(1, len(ordered) + 1))
        rankings.append(ordered)
        required = 2 * k if group == "both" else k
        if k < 1 or len(ordered) < required:
            raise ValueError(f"{task}: insufficient agents for {group} K={k}")
        if group in {"top", "both"}:
            selections.append(ordered.head(k).assign(group="top"))
        if group in {"bottom", "both"}:
            bottom = ordered.sort_values(["mean_return", "csv_idx", "policy_seed"], ascending=[True, True, True])
            # Both groups must be disjoint even if all performance values tie.
            if group == "both":
                top_keys = pd.MultiIndex.from_frame(ordered.head(k)[AGENT_KEYS])
                bottom = bottom.loc[~pd.MultiIndex.from_frame(bottom[AGENT_KEYS]).isin(top_keys)]
            selections.append(bottom.head(k).assign(group="bottom"))
    return pd.concat(rankings, ignore_index=True), pd.concat(selections, ignore_index=True)


def aligned_episode(data, protocol, observation_dim, action_dim, action_type):
    """Adapt old corrected NPZs and validate modern post-update samples.

    Legacy input action columns encode zeros, so old legacy arrays cannot recover
    executed actions. Return None to request checkpoint recollection.
    """
    if "contexts" in data and all(key in data for key in SIGNALS):
        result = {key: np.asarray(data[key]) for key in ("contexts", *SIGNALS)}
    elif protocol == "corrected" and "states" in data and "inputs" in data:
        states, inputs = np.asarray(data["states"]), np.asarray(data["inputs"])
        if len(states) != len(inputs) or len(states) < 2:
            raise ValueError("Invalid corrected state/input boundary alignment")
        inputs = inputs[:-1]
        encoded = inputs[:, observation_dim : observation_dim + action_dim]
        result = {
            "contexts": states[1:],
            "observations": inputs[:, :observation_dim],
            "actions": encoded.argmax(axis=1) if action_type == "discrete" else encoded,
            "rewards": inputs[:, -1],
        }
    else:
        return None
    contexts = result["contexts"]
    n = len(contexts)
    if contexts.ndim != 2 or n == 0 or contexts.shape[1] == 0:
        raise ValueError("Contexts must be a nonempty samples × neurons matrix")
    for key, value in result.items():
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"Unaligned or nonfinite {key}")
    if result["observations"].shape != (n, observation_dim) or result["rewards"].shape != (n,):
        raise ValueError("Invalid observation or reward dimensions")
    if action_type == "discrete":
        actions = result["actions"].reshape(-1)
        if not np.equal(actions, actions.astype(int)).all() or (actions < 0).any() or (actions >= action_dim).any():
            raise ValueError("Invalid categorical action labels")
        result["actions"] = actions.astype(np.int64)
    elif result["actions"].shape != (n, action_dim):
        raise ValueError("Invalid continuous action dimensions")
    result["timesteps"] = np.arange(n)
    result["action_type"] = action_type
    return result


def cached_agent(args, record, trajectory_registry, env):
    """Reuse complete, verified episodes; recollect a whole agent if any are absent."""
    if trajectory_registry is None:
        return None
    rows = agent_rows(trajectory_registry, record).sort_values("evaluation_seed")
    seeds = list(range(args.evaluation_seed, args.evaluation_seed + args.evaluation_episodes))
    if rows.evaluation_seed.tolist() != seeds:
        return None
    wrapper = env.env
    action_dim = wrapper.action_space.n if wrapper.discrete_a_space else int(np.prod(wrapper.action_space.shape))
    action_type = "discrete" if wrapper.discrete_a_space else "continuous"
    observation_dim = int(np.prod(wrapper.env.observation_space.shape))
    trials = []
    for _, trial in rows.iterrows():
        if trial.reservoir_seed != record.reservoir_seed:
            raise ValueError("Trajectory reservoir seed mismatch")
        for key, actual in (
            ("matrix_hash", matrix_hash(wrapper.reservoir.W)),
            ("readin_hash", matrix_hash(wrapper.reservoir.Win)),
        ):
            if trial.get(key, "") and trial[key] != actual:
                raise ValueError(f"Trajectory {key} mismatch")
        if trial.get("model_sha256", "") and trial.model_sha256 != file_hash(record.model_path):
            raise ValueError("Trajectory checkpoint fingerprint mismatch")
        if trial.get("encoder_sha256", "") and trial.encoder_sha256 != record.encoder_sha256:
            raise ValueError("Trajectory encoder fingerprint mismatch")
        path = Path(trial.path)
        if not path.is_absolute():
            path = args.trajectories_csv.parent / path
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            for key in (*EPISODE_KEYS, "protocol", "reservoir_seed", "selection_sha256"):
                if key not in data or data[key].item() != trial[key]:
                    raise ValueError(f"Trajectory NPZ {key} disagrees with registry")
            for key, actual in (
                ("matrix_hash", matrix_hash(wrapper.reservoir.W)),
                ("readin_hash", matrix_hash(wrapper.reservoir.Win)),
                ("encoder_sha256", record.get("encoder_sha256", "")),
                ("model_sha256", file_hash(record.model_path)),
            ):
                if key in data and data[key].item() != actual:
                    raise ValueError(f"Trajectory NPZ {key} mismatch")
            aligned = aligned_episode(data, args.protocol, observation_dim, action_dim, action_type)
        if aligned is None:
            return None
        if aligned["contexts"].shape[1] != wrapper.reservoir.units:
            raise ValueError("Trajectory context size does not match reservoir")
        trials.append(
            {
                **aligned,
                "episode": int(trial.episode),
                "evaluation_seed": int(trial.evaluation_seed),
                "source_path": str(path.resolve()),
            }
        )
    return trials


def load_agent_trials(args, frame, record, registry, output):
    """Reconstruct the encoder, then reuse or collect its aligned episodes."""
    config, row = registry_environment(record, args, frame)
    if config["protocol"] != args.protocol:
        raise ValueError("Checkpoint environment protocol mismatch")
    env = make_environment(record.task, config, row, args.selected_csv, False, args.evaluation_seed)
    try:
        for key, actual in (
            ("matrix_hash", matrix_hash(env.env.reservoir.W)),
            ("readin_hash", matrix_hash(env.env.reservoir.Win)),
        ):
            if record.get(key, "") and record[key] != actual:
                raise ValueError(f"Checkpoint {key} mismatch")
        trials = cached_agent(args, record, registry, env)
        if trials is not None:
            return trials
    finally:
        env.close()
    print(f"Collecting {record.task} reservoir {record.csv_idx}, policy {record.policy_seed}", flush=True)
    # Reuse the standard collector; give each agent its own staging directory so
    # its episode table cannot overwrite the complete ranking table.
    staging = output / "trajectories" / agent_id(record)
    for folder in ("evaluation", "trajectories"):
        (staging / folder).mkdir(parents=True, exist_ok=True)
    evaluate_registry(args, frame, pd.DataFrame([record]), staging, collect=True)
    collected = pd.read_csv(staging / "trajectories" / "trajectories.csv", keep_default_na=False)
    env = make_environment(record.task, config, row, args.selected_csv, False, args.evaluation_seed)
    try:
        return cached_agent(args, record, collected, env)
    finally:
        env.close()


def agent_id(record):
    """Stable filename-safe identifier for the supported Gym task names."""
    task = str(record["task"]).replace("/", "_").replace("\\", "_")
    return f"{task}_idx{int(record['csv_idx'])}_policy{int(record['policy_seed'])}"


def combine_trials(trials):
    """Concatenate aligned episodes while retaining their boundaries."""
    if not trials:
        raise ValueError("No aligned episode data")
    result = {key: np.concatenate([trial[key] for trial in trials]) for key in ("contexts", *SIGNALS, "timesteps")}
    result["episodes"] = np.concatenate([np.full(len(t["contexts"]), t["episode"]) for t in trials])
    result["evaluation_seeds"] = np.concatenate([np.full(len(t["contexts"]), t["evaluation_seed"]) for t in trials])
    result["action_type"] = trials[0]["action_type"]
    return result


def fit_pca(contexts):
    """Center activity without rescaling individual neuron variances."""
    if contexts.ndim != 2 or min(contexts.shape) < 3 or not np.isfinite(contexts).all():
        raise ValueError("PCA needs at least three samples and neurons, all finite")
    if not np.any(np.ptp(contexts, axis=0) > 0):
        raise ValueError("PCA activity is constant")
    model = PCA(n_components=3, svd_solver="full")
    return model, model.fit_transform(contexts)


def fit_cebra(contexts, labels, discrete, args):
    """Fit one behavior-only embedding, skipping constant auxiliary signals."""
    import torch

    import cebra

    labels = np.asarray(labels)
    if (
        contexts.ndim != 2
        or len(contexts) < 3
        or len(labels) != len(contexts)
        or not np.isfinite(contexts).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("CEBRA needs aligned finite activity and labels, with at least three samples")
    if not np.any(np.ptp(labels, axis=0) > 0):
        return None, None, None
    np.random.seed(args.analysis_seed)
    torch.manual_seed(args.analysis_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.analysis_seed)
    scaler = None
    if discrete:
        labels = labels.astype(np.int64).reshape(-1)
    else:
        scaler = StandardScaler()
        labels = scaler.fit_transform(labels.reshape(len(labels), -1)).astype(np.float32)
    model = cebra.CEBRA(
        model_architecture="offset1-model",
        output_dimension=3,
        conditional=None if discrete else "delta",
        delta=args.cebra_delta,
        distance="cosine",
        batch_size=args.cebra_batch_size,
        learning_rate=args.cebra_learning_rate,
        max_iterations=args.cebra_iterations,
        device=args.device,
        verbose=True,
    )
    model.fit(contexts.astype(np.float32), labels)
    embedding = model.transform(contexts.astype(np.float32))
    if embedding.shape != (len(contexts), 3) or not np.isfinite(embedding).all():
        raise ValueError("CEBRA returned invalid embedding")
    return model, embedding, scaler


def plot_views(panels, target, title, dpi):
    """Plot 3D and two 2D views; each independent fit gets a separate row."""
    from matplotlib.colors import Normalize

    plt = pyplot()
    fig = plt.figure(figsize=(17, 5 * len(panels)), layout="constrained")
    grid = fig.add_gridspec(len(panels), 3)
    fig.suptitle(title)
    for row, panel in enumerate(panels):
        axes = [
            fig.add_subplot(grid[row, 0], projection="3d"),
            fig.add_subplot(grid[row, 1]),
            fig.add_subplot(grid[row, 2]),
        ]
        if panel.get("embedding") is None:
            for axis in axes:
                text = axis.text2D if hasattr(axis, "text2D") else axis.text
                text(0.1, 0.5, panel["label"] + "\nConstant signal: skipped", transform=axis.transAxes)
            continue
        embedding, rewards = panel["embedding"], panel["rewards"]
        names = panel["axes"]
        norm = Normalize(vmin=float(rewards.min()), vmax=float(rewards.max()))
        for j, axis in enumerate(axes):
            for agent in np.unique(panel["agent_index"]):
                mask = panel["agent_index"] == agent
                coordinates = [embedding[mask, 0], embedding[mask, 1 if j < 2 else 2]]
                if j == 0:
                    coordinates.append(embedding[mask, 2])
                image = axis.scatter(
                    *coordinates,
                    c=rewards[mask],
                    cmap="viridis",
                    norm=norm,
                    s=5,
                    alpha=0.65,
                    marker=("o", "^", "s", "D", "v", "P", "X")[int(agent) % 7],
                    label=panel["agent_labels"][int(agent)],
                )
            axis.set_xlabel(names[0])
            axis.set_ylabel(names[1 if j < 2 else 2])
            if j == 0:
                axis.set_zlabel(names[2])
            axis.set_title(panel["label"])
            if len(panel["agent_labels"]) > 1:
                axis.legend(fontsize=6)
            fig.colorbar(image, ax=axis, shrink=0.65, pad=0.12 if j == 0 else 0.02, label="Step reward")
    fig.savefig(target, dpi=dpi)
    plt.close(fig)


def embed_task(args, task, agents, output):
    """Fit independent or explicitly pooled agent embeddings for one task."""
    if args.embedding_mode == "shared":
        dimensions = {
            (
                a["data"]["contexts"].shape[1],
                a["data"]["observations"].shape[1],
                a["data"]["actions"].shape[1:],
                a["data"]["action_type"],
            )
            for a in agents
        }
        if len(dimensions) != 1:
            raise ValueError("Shared embeddings require matching neuron and label dimensions")
        groups = [agents]
    else:
        groups = [[a] for a in agents]
    panels = {key: [] for key in ("pca", *SIGNALS)}
    statuses = []
    for group in groups:
        identifier = f"{task}_shared" if args.embedding_mode == "shared" else agent_id(group[0]["record"])
        contexts = np.concatenate([a["data"]["contexts"] for a in group])
        rewards = np.concatenate([a["data"]["rewards"] for a in group])
        indices = np.concatenate([np.full(len(a["data"]["contexts"]), i) for i, a in enumerate(group)])
        labels = [
            f"{a['record'].group} rank {a['record']['rank']} idx {a['record'].csv_idx} "
            f"policy {a['record'].policy_seed}, return {a['record'].mean_return:.2f}"
            for a in group
        ]
        base = {
            "rewards": rewards,
            "agent_index": indices,
            "agent_labels": labels,
            "label": "Shared agents" if len(group) > 1 else labels[0],
        }
        pca, embedding = fit_pca(contexts)
        joblib.dump(pca, output / "models" / f"{identifier}_pca.joblib")
        variance = pca.explained_variance_ratio_
        panels["pca"].append(
            {**base, "embedding": embedding, "axes": [f"PC{i + 1} ({v:.1%})" for i, v in enumerate(variance)]}
        )
        arrays = {
            "pca": embedding,
            "explained_variance_ratio": variance,
            "agent_index": indices,
            "agent_labels": np.asarray(labels),
        }
        for key in ("episodes", "evaluation_seeds", "timesteps", *SIGNALS):
            arrays[key] = np.concatenate([a["data"][key] for a in group])
        for signal in SIGNALS:
            print(f"CEBRA {identifier}: {signal}", flush=True)
            model, embedded, scaler = fit_cebra(
                contexts, arrays[signal], signal == "actions" and group[0]["data"]["action_type"] == "discrete", args
            )
            panels[signal].append({**base, "embedding": embedded, "axes": ["CEBRA 1", "CEBRA 2", "CEBRA 3"]})
            statuses.append(
                {
                    "embedding": identifier,
                    "signal": signal,
                    "status": "constant signal: skipped" if model is None else "ok",
                }
            )
            if model is None:
                continue
            arrays[f"cebra_{signal}"] = embedded
            model.save(str(output / "models" / f"{identifier}_cebra_{signal}.pt"))
            if scaler is not None:
                joblib.dump(scaler, output / "models" / f"{identifier}_{signal}_scaler.joblib")
            losses = np.asarray(model.state_dict_["loss"])
            arrays[f"loss_{signal}"] = losses
            plt = pyplot()
            fig, axis = plt.subplots(figsize=(6, 3))
            axis.plot(losses)
            axis.set(xlabel="Iteration", ylabel="InfoNCE loss", title=f"{identifier}: {signal}")
            fig.savefig(output / "figures" / f"{identifier}_{signal}_loss.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
        np.savez_compressed(output / "analysis" / f"{identifier}_embeddings.npz", **arrays)
    for signal, rows in panels.items():
        plot_views(
            rows,
            output / "figures" / f"{task}_{args.embedding_mode}_{signal}.png",
            f"{task}: {'PCA' if signal == 'pca' else 'CEBRA / ' + signal}",
            args.dpi,
        )
    return statuses


def run_embeddings(args, frame, output):
    """Evaluate missing policies, select K agents, collect labels, and plot."""
    registry = validate_sources(args, frame)
    if args.evaluation_csv:
        episodes = read_evaluation(args.evaluation_csv, args, registry)
    else:
        episodes = pd.DataFrame(columns=EPISODE_KEYS)
    present = pd.MultiIndex.from_frame(episodes[AGENT_KEYS])
    missing = registry.loc[~pd.MultiIndex.from_frame(registry[AGENT_KEYS]).isin(present)]
    if not missing.empty:
        print(f"Evaluating {len(missing)} agents × {args.evaluation_episodes} common seeded episodes", flush=True)
        evaluated = evaluate_registry(args, frame, missing, output)
        episodes = evaluated if episodes.empty else pd.concat([episodes, evaluated], ignore_index=True)
    episodes.to_csv(output / "evaluation" / "episodes.csv", index=False)
    rankings, selected = select_agents(episodes, args.k, args.group)
    rankings.to_csv(output / "analysis" / "rankings.csv", index=False)
    rankings.to_csv(output / "evaluation" / "returns.csv", index=False)
    selected = selected.merge(registry, on=AGENT_KEYS, validate="one_to_one")
    selected.to_csv(output / "analysis" / "selected_agents.csv", index=False)
    saved = read_trajectory_registry(args.trajectories_csv, args) if args.trajectories_csv else None
    provenance, trajectories, statuses = [], [], []
    for task in args.tasks:
        agents = []
        for _, record in selected[selected.task == task].iterrows():
            trials = load_agent_trials(args, frame, record, saved, output)
            data = combine_trials(trials)
            config, _ = registry_environment(record, args, frame)
            sources = [t["source_path"] for t in trials]
            provenance.append(
                {
                    "agent": agent_id(record),
                    "configuration": config,
                    "model_sha256": file_hash(record.model_path),
                    "sources": [{"path": p, "sha256": file_hash(p)} for p in sources],
                }
            )
            # Export a flat registry referencing verified source files for reuse.
            for trial in trials:
                with np.load(trial["source_path"], allow_pickle=False) as original:
                    common = {
                        key: original[key].item()
                        for key in (
                            *AGENT_KEYS,
                            "episode",
                            "evaluation_seed",
                            "protocol",
                            "selection_sha256",
                            "reservoir_seed",
                        )
                    }
                    hashes = {
                        key: original[key].item() if key in original else record.get(key, "")
                        for key in ("matrix_hash", "readin_hash")
                    }
                trajectories.append(
                    {
                        **common,
                        "path": trial["source_path"],
                        "stratum": record.stratum,
                        **hashes,
                        "model_sha256": file_hash(record.model_path),
                        "encoder_sha256": record.encoder_sha256,
                    }
                )
            np.savez_compressed(output / "trajectories" / f"{agent_id(record)}_aligned.npz", **data)
            agents.append({"record": record, "data": data})
        statuses.extend(embed_task(args, task, agents, output))
        write_json(output / "analysis" / "provenance.json", provenance)
        write_json(output / "analysis" / "embedding_status.json", statuses)
        pd.DataFrame(trajectories).to_csv(output / "trajectories" / "trajectories.csv", index=False)
    return {
        "agent_count": len(registry),
        "selected_agent_count": len(selected),
        "embedding_mode": args.embedding_mode,
        "embedding_status": statuses,
        "alignment": "post-update context, resulting observation, executed action, transition reward",
    }
