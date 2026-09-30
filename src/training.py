"""PPO training with separate structure and optimization seeds.

Training writes models, TensorBoard events, and a models.csv registry into an
isolated run. Legacy checkpoint discovery remains read-only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reservoirs import load_reservoir, reservoir_path
from src.utils import configure_tensorboard, experiment_name, file_hash, legacy_config, matrix_hash, write_json


def make_environment(task, config, row=None, selected_csv=None, baseline=False, environment_seed=0):
    """Build an equally masked baseline or context encoder for a fixed protocol."""
    configure_tensorboard()
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor

    from src.er_mrl.wrappers import DeletedVelocityWrapper, ReservoirWrapper

    env = (
        gym.make(task, max_episode_steps=config["max_episode_steps"])
        if config.get("max_episode_steps")
        else gym.make(task)
    )
    try:
        if config["del_obs"]:
            env = DeletedVelocityWrapper(env)
        if not baseline:
            if config["protocol"] == "corrected":
                artifact = load_reservoir(reservoir_path(selected_csv, row), row, config["units"], config["res_sr"])
                kwargs = {"seed": int(row["seed"]), "recurrent_matrix": artifact["W"]}
            elif row is not None:
                kwargs = legacy_config(row)
                kwargs.pop("csv_idx")
                kwargs.pop("core_fraction")
                # Historical single-motif training varied the reservoir with policy seed.
                kwargs["seed"] = config.get("reservoir_seed", int(row["seed"]))
            else:
                kwargs = {
                    key: config[key]
                    for key in (
                        "motif",
                        "n_communities",
                        "hi",
                        "lo",
                        "mid",
                        "sigma",
                        "connectivity",
                        "symmetric",
                        "p_negative",
                    )
                }
                kwargs["seed"] = config["reservoir_seed"]
            env = ReservoirWrapper(
                env,
                units=config["units"],
                lr=config["res_lr"],
                sr=config["res_sr"],
                iss=config["res_iss"],
                skip_c=config["skip_c"],
                reset_res=config["reset_res"],
                protocol=config["protocol"],
                **kwargs,
            )
            # Initialize weights without feeding a warm-up observation into legacy state.
            actions = env.action_space.n if env.discrete_a_space else int(np.prod(env.action_space.shape))
            features = int(np.prod(env.env.observation_space.shape)) + actions + 1
            env.reservoir.initialize(np.zeros(features))
        env.action_space.seed(environment_seed)
        return Monitor(env)
    except Exception:
        env.close()
        raise


def training_config(args):
    """Extract serializable environment and policy settings from parsed options."""
    keys = (
        "protocol",
        "units",
        "res_lr",
        "res_sr",
        "res_iss",
        "skip_c",
        "reset_res",
        "del_obs",
        "motif",
        "n_communities",
        "hi",
        "lo",
        "mid",
        "sigma",
        "connectivity",
        "symmetric",
        "p_negative",
        "max_episode_steps",
        "learning_rate",
        "training_steps",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "device",
        "reservoir_seed",
        "environment_seed",
    )
    return {key: getattr(args, key) for key in keys}


def train_one(task, row, policy_seed, config, output, selected_csv=None, baseline=False, indexed=True):
    """Train one PPO replicate and return a registry row; always close environments."""
    configure_tensorboard()
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    set_random_seed(policy_seed)
    motif = None if baseline else row["stratum"] if row is not None else config["motif"]
    index = int(row["csv_idx"]) if row is not None and indexed else None
    name, prefix = experiment_name(task, motif, config["training_steps"], index)
    identifier = f"{name}_policy{policy_seed}"
    model_path = output / "models" / f"{identifier}.zip"
    logdir = output / "logs" / identifier
    local_config = dict(config)
    if local_config["reservoir_seed"] is None:
        local_config["reservoir_seed"] = int(row["seed"]) if row is not None else policy_seed
    factories = []
    for rank in range(config["n_envs"]):

        def factory(rank=rank):
            """Create a worker with fixed structure and an independently seeded simulator."""
            return make_environment(task, local_config, row, selected_csv, baseline, config["environment_seed"] + rank)

        factories.append(factory)
    vector_cls = DummyVecEnv if config["n_envs"] == 1 else SubprocVecEnv
    env = vector_cls(factories)
    try:
        # Seed simulator state independently; PPO normally seeds it with its own seed.
        model = PPO(
            "MlpPolicy",
            env,
            seed=policy_seed,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            tensorboard_log=str(logdir),
            device=config["device"],
            verbose=1,
        )
        env.seed(config["environment_seed"] if config["protocol"] == "corrected" else policy_seed)
        model.learn(total_timesteps=config["training_steps"], tb_log_name=f"{prefix}_seed_{policy_seed}")
        model.save(model_path)
        reservoir_hash = None
        readin_hash = ""
        if not baseline:
            reservoirs = env.get_attr("reservoir")
            hashes = {matrix_hash(reservoir.W) for reservoir in reservoirs}
            readin_hashes = {matrix_hash(reservoir.Win) for reservoir in reservoirs}
            if len(hashes) != 1 or len(readin_hashes) != 1:
                raise ValueError("Parallel environments do not share identical reservoir/read-in weights")
            reservoir_hash, readin_hash = hashes.pop(), readin_hashes.pop()
            if config["protocol"] == "corrected":
                if reservoir_hash != str(load_reservoir(reservoir_path(selected_csv, row))["matrix_hash"]):
                    raise ValueError("Training matrix changed after selection")
        record = {
            "task": task,
            "csv_idx": int(row["csv_idx"]) if row is not None and not baseline else -1,
            "stratum": motif or "PPO",
            "policy_seed": policy_seed,
            "reservoir_seed": local_config["reservoir_seed"] if not baseline else -1,
            "model_path": str(model_path.resolve()),
            "log_path": str(logdir.resolve()),
            "protocol": config["protocol"],
            "matrix_hash": reservoir_hash or "",
            "readin_hash": readin_hash,
            "selected_csv": str(Path(selected_csv).resolve()) if selected_csv else "",
            "selection_sha256": file_hash(selected_csv) if selected_csv else "",
            "config_path": str((output / "models" / f"{identifier}.json").resolve()),
        }
        write_json(
            record["config_path"],
            {"environment": local_config, "row": row.to_dict() if row is not None else None, "record": record},
        )
        return record
    finally:
        env.close()


def train_batch(args, frame, output, first_per_stratum=False, standalone=False):
    """Train requested configurations and baseline, checkpointing the registry each run."""
    config = training_config(args)
    if first_per_stratum and frame is not None:
        frame = frame.drop_duplicates("stratum")
    if args.limit is not None and frame is not None:
        frame = frame.head(args.limit)
    records = []
    for task in args.tasks:
        if args.baseline or (standalone and not args.use_reservoir):
            for seed in args.policy_seeds:
                records.append(train_one(task, None, seed, config, output, baseline=True))
                pd.DataFrame(records).to_csv(output / "models" / "models.csv", index=False)
        if not args.use_reservoir:
            continue
        rows = (
            [None]
            if standalone and args.protocol == "legacy" and getattr(args, "csv_idx", None) is None
            else [row for _, row in frame.iterrows()]
        )
        for row in rows:
            for seed in args.policy_seeds:
                local = dict(config)
                actual_seed = seed
                if args.protocol == "legacy" and row is not None and not standalone:
                    actual_seed = int(row["seed"]) + seed
                    local["reservoir_seed"] = actual_seed
                records.append(
                    train_one(task, row, actual_seed, local, output, args.selected_csv, indexed=not first_per_stratum)
                )
                pd.DataFrame(records).to_csv(output / "models" / "models.csv", index=False)
    return pd.DataFrame(records)


def discover_legacy_models(frame, tasks, models_dir, logs_dir, steps):
    """Read existing all-selected checkpoints without modifying their directories."""
    records = []
    for task in tasks:
        for _, row in frame.iterrows():
            name, prefix = experiment_name(task, row.stratum, steps, int(row.csv_idx))
            model = Path(models_dir) / name / f"{prefix}_model_seed_{int(row.seed)}.zip"
            if model.exists():
                records.append(
                    {
                        "task": task,
                        "csv_idx": int(row.csv_idx),
                        "stratum": row.stratum,
                        "policy_seed": int(row.seed),
                        "reservoir_seed": int(row.seed),
                        "model_path": str(model.resolve()),
                        "log_path": str((Path(logs_dir) / name).resolve()),
                        "protocol": "legacy",
                        "config_path": "",
                        "matrix_hash": "",
                        "selection_sha256": "",
                    }
                )
    return pd.DataFrame(records)


def load_registry(args, frame):
    """Read a model registry and reject selection or protocol mismatches."""
    if args.models_csv:
        registry = pd.read_csv(args.models_csv, keep_default_na=False)
        required = {"task", "csv_idx", "policy_seed", "model_path", "protocol", "config_path"}
        if required - set(registry):
            raise ValueError(f"Model registry missing columns: {sorted(required - set(registry))}")
        if not registry.protocol.eq(args.protocol).all():
            raise ValueError("Model registry protocol does not match --protocol")
        if "selection_sha256" in registry:
            hashes = registry.loc[registry.csv_idx >= 0, "selection_sha256"]
            if hashes.ne(file_hash(args.selected_csv)).any():
                raise ValueError("Model registry selection fingerprint mismatch")
        registry = registry[registry.task.isin(args.tasks)]
    elif args.protocol == "legacy":
        registry = discover_legacy_models(frame, args.tasks, args.models_dir, args.logs_dir, args.training_steps)
    else:
        raise ValueError("Corrected evaluation requires --models-csv")
    if registry.empty:
        raise ValueError("No models found for requested tasks")
    if args.limit is not None:
        ids = frame.csv_idx.head(args.limit)
        registry = registry[(registry.csv_idx < 0) | registry.csv_idx.isin(ids)]
    return registry


def registry_environment(record, args, frame):
    """Recover saved environment configuration; legacy artifacts use explicit CLI settings."""
    row = None if int(record.csv_idx) < 0 else frame.set_index("csv_idx", drop=False).loc[int(record.csv_idx)]
    if record.config_path:
        metadata = json.loads(Path(record.config_path).read_text())
        config = metadata["environment"]
        if metadata["row"] is None:
            row = None
    else:
        config = training_config(args)
        config["reservoir_seed"] = int(record.reservoir_seed)
    return config, row
