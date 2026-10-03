"""Strict cohort, checkpoint, episode and cache validation for Solution 1."""

import hashlib
import importlib.metadata
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import METRICS, aggregate_policy_distances, fit_policy_trajectories
from src.utils import ROOT, STRATA, TASKS, file_hash, imported_artifact_path, read_scalars, write_json

KEYS = ["task", "csv_idx", "policy_seed", "episode"]
PPO_KEYS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "environment_seed",
)
ENCODER_KEYS = (
    "protocol",
    "units",
    "res_lr",
    "res_sr",
    "res_iss",
    "del_obs",
    "skip_c",
    "reset_res",
    "reservoir_seed",
    "max_episode_steps",
)


def manifest_for(path, selection_hash):
    """Require a completed collection manifest belonging to this frozen selection."""
    target = Path(path).parent.parent / "manifest.json"
    value = json.loads(target.read_text())
    if value.get("status") != "completed" or value.get("selection_sha256") != selection_hash:
        raise ValueError(f"Incomplete collection or selection fingerprint mismatch: {target}")
    return target, value


def episode_cohort(table, frame, args, dynamics=False):
    """Require exactly the specified episode schedule for every requested policy."""
    required = {*KEYS, "evaluation_seed", "protocol", "selection_sha256", "model_sha256", "encoder_sha256"}
    if required - set(table) or not table.protocol.eq("corrected").all():
        raise ValueError(f"Episode table missing columns/protocol: {sorted(required - set(table))}")
    relevant = table[
        table.task.isin(args.tasks) & table.csv_idx.isin(frame.csv_idx) & table.policy_seed.isin(args.policy_seeds)
    ].copy()
    count = args.evaluation_episodes if dynamics else 10
    if args.smoke:
        count = args.evaluation_episodes
        relevant = relevant[relevant.episode < count]
    if (
        relevant.duplicated(KEYS).any()
        or len(relevant) != len(args.tasks) * len(frame) * len(args.policy_seeds) * count
    ):
        raise ValueError("Missing or duplicate reservoir-policy episodes; complete task cohorts are required")
    digest = file_hash(args.selected_csv)
    if not relevant.selection_sha256.eq(digest).all():
        raise ValueError("Episode selection fingerprint mismatch")
    expected_seeds = set(range(20000 if dynamics else 10000, (20000 if dynamics else 10000) + count))
    for _, rows in relevant.groupby(["task", "csv_idx", "policy_seed"]):
        if set(rows.episode) != set(range(count)):
            raise ValueError("Incomplete episode IDs")
        if set(rows.evaluation_seed) != expected_seeds:
            if not (
                dynamics
                and args.exploratory_reuse_ranking_episodes
                and set(rows.evaluation_seed) == set(range(10000, 10000 + count))
            ):
                raise ValueError("Ranking requires seeds 10000–10009; dynamics requires separate seeds 20000–20009")
        if (rows.evaluation_seed.to_numpy() - rows.episode.to_numpy() != min(rows.evaluation_seed)).any():
            raise ValueError("Episode/environment seed mapping mismatch")
        if rows.model_sha256.nunique() != 1 or rows.encoder_sha256.nunique() != 1:
            raise ValueError("A policy's episodes use different checkpoints or encoders")
    return relevant


def validate_design(args, frame):
    """Enforce the fixed design; smoke runs and explicitly exploratory fits are labelled."""
    if args.allow_subset:
        raise ValueError("Solution 1 cannot drop reservoirs/policies; --interim permits only complete task subsets")
    args.policy_seeds = sorted(args.policy_seeds)
    if not set(args.tasks).issubset(TASKS) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Solution 1 requires unique paper task IDs")
    if set(args.tasks) != set(TASKS) and not (args.interim or args.smoke):
        raise ValueError("Use --interim for completed tasks before all five are available")
    if not args.smoke:
        if len(frame) != 20 or frame.stratum.value_counts().to_dict() != dict.fromkeys(STRATA, 4):
            raise ValueError("Solution 1 requires all 20 reservoirs, four per family")
        if frame.matrix_hash.nunique() != 20:
            raise ValueError("Solution 1 requires 20 distinct recurrent matrices")
        if set(args.policy_seeds) != set(range(5)) or not frame.units.eq(200).all():
            raise ValueError("Solution 1 requires five policy seeds 0–4 and 200 neurons")
        if args.group_size != 5 or args.ranking_source != "evaluation":
            raise ValueError("Solution 1 uses fixed top/bottom five and deterministic ranking episodes")
        settings = (
            args.backend in ("dmdc", "n4sid")
            and args.n_delays == 3
            and args.rank is None
            and args.rank_energy == 0.99
            and args.max_rank == 50
            and args.min_rank == 1
            and args.dmd_regularization == 1e-8
            and args.evaluation_episodes == 10
            and args.analysis_seed == 0
            and args.permutations == 9999
            and args.bootstrap_replicates == 5000
        )
        if not settings and not (args.exploratory_reuse_ranking_episodes or args.exploratory_identification):
            raise ValueError("Changed Solution 1 identification/inference settings require explicit exploratory mode")
        selection_manifest = json.loads((Path(args.selected_csv).parent.parent / "manifest.json").read_text())
        settings = selection_manifest["arguments"]
        if (
            selection_manifest.get("status") != "completed"
            or settings.get("pca_components") != 8
            or settings.get("pool_per_stratum") != 1000
            or settings.get("select_per_stratum") != 4
            or settings.get("selection_seed") != 0
        ):
            raise ValueError("Selection does not match the frozen Solution 1 design")
        pool = pd.read_csv(args.pool_csv, keep_default_na=False)
        if len(pool) != 5000 - selection_manifest["rejected_count"]:
            raise ValueError("Descriptor standardization requires the full valid candidate pool")


def validate_sources(args, frame, output):
    """Audit imported files without changing registries; return complete local source data."""
    validate_design(args, frame)
    if args.evaluation_csv is None or args.trajectories_csv is None:
        raise ValueError("Solution 1 requires ranking --evaluation-csv and separate dynamics --trajectories-csv")
    selection_hash = file_hash(args.selected_csv)
    ranking_manifest_path, ranking_manifest = manifest_for(args.evaluation_csv, selection_hash)
    dynamics_manifest_path, _ = manifest_for(args.trajectories_csv, selection_hash)
    source_hashes = {
        str(p): file_hash(p)
        for p in (
            args.selected_csv,
            args.pool_csv,
            args.evaluation_csv,
            args.trajectories_csv,
            ranking_manifest_path,
            dynamics_manifest_path,
        )
    }
    ranking = episode_cohort(pd.read_csv(args.evaluation_csv, keep_default_na=False), frame, args)
    dynamics = episode_cohort(pd.read_csv(args.trajectories_csv, keep_default_na=False), frame, args, dynamics=True)
    ranking_sets = ranking.groupby(["task", "csv_idx", "policy_seed"]).evaluation_seed.apply(set)
    for key, rows in dynamics.groupby(["task", "csv_idx", "policy_seed"]):
        if ranking_sets.loc[key] & set(rows.evaluation_seed) and not args.exploratory_reuse_ranking_episodes:
            raise ValueError("Ranking and dynamics episodes overlap; recollect dynamics with --evaluation-seed 20000")
    if not np.isfinite(ranking["return"].to_numpy(float)).all():
        raise ValueError("Nonfinite ranking returns")
    dynamics_episodes_path = Path(args.trajectories_csv).parent.parent / "evaluation" / "episodes.csv"
    dynamics_episodes = episode_cohort(pd.read_csv(dynamics_episodes_path, keep_default_na=False), frame, args, True)
    source_hashes[str(dynamics_episodes_path)] = file_hash(dynamics_episodes_path)
    paired = dynamics.merge(dynamics_episodes, on=KEYS, validate="one_to_one", suffixes=("", "_eval"))
    for field in ("evaluation_seed", "model_sha256", "encoder_sha256", "selection_sha256", "reservoir_seed"):
        if not paired[field].eq(paired[field + "_eval"]).all():
            raise ValueError(f"Dynamics episode and trajectory tables disagree: {field}")
    if dynamics.groupby(["task", "csv_idx"]).readin_hash.nunique().ne(1).any() or dynamics.readin_hash.eq("").any():
        raise ValueError("Read-in matrix must be fixed across policy seeds within each reservoir/task")
    matrix_hashes = frame.set_index("csv_idx").matrix_hash
    if not dynamics.matrix_hash.eq(dynamics.csv_idx.map(matrix_hashes)).all():
        raise ValueError("Dynamics recurrent matrix identity mismatch")
    models_path = args.models_csv
    if models_path is None:
        models_path = imported_artifact_path(ranking_manifest["arguments"]["models_csv"], args.evaluation_csv)
    args.models_csv = Path(models_path)
    source_hashes[str(models_path)] = file_hash(models_path)
    training_manifest_path = Path(models_path).parent.parent / "manifest.json"
    training_history = None
    if training_manifest_path.exists():
        training_manifest = json.loads(training_manifest_path.read_text())
        source_hashes[str(training_manifest_path)] = file_hash(training_manifest_path)
        training_history = dict(
            arguments=training_manifest.get("arguments"), resume_history=training_manifest.get("resume_history", [])
        )
    models = pd.read_csv(models_path, keep_default_na=False)
    models = models[
        models.task.isin(args.tasks) & models.csv_idx.isin(frame.csv_idx) & models.policy_seed.isin(args.policy_seeds)
    ].copy()
    if models.duplicated(["task", "csv_idx", "policy_seed"]).any() or len(models) != len(frame) * len(args.tasks) * len(
        args.policy_seeds
    ):
        raise ValueError("Missing or duplicate trained reservoir policies")
    audits, configurations, curves = [], [], {}
    print(f"Validating {len(models)} checkpoints and {len(dynamics)} dynamics episodes", flush=True)
    for record in models.itertuples():
        key = (record.task, record.csv_idx, record.policy_seed)
        model_path = imported_artifact_path(record.model_path, models_path)
        config_path = imported_artifact_path(record.config_path, models_path)
        log_path = imported_artifact_path(record.log_path, models_path)
        metadata = json.loads(config_path.read_text())
        config = metadata["environment"]
        with zipfile.ZipFile(model_path) as archive:
            saved = json.loads(archive.read("data"))
        digest = file_hash(model_path)
        encoder_hash = hashlib.sha256(
            json.dumps({k: config.get(k) for k in ENCODER_KEYS}, sort_keys=True).encode()
        ).hexdigest()
        for table in (ranking, dynamics):
            rows = table[(table.task == key[0]) & (table.csv_idx == key[1]) & (table.policy_seed == key[2])]
            if not rows.model_sha256.eq(digest).all() or not rows.encoder_sha256.eq(encoder_hash).all():
                raise ValueError(f"Checkpoint/encoder differs between training, ranking and dynamics: {key}")
        if (
            record.protocol != "corrected"
            or record.selection_sha256 != selection_hash
            or record.matrix_hash != matrix_hashes.loc[record.csv_idx]
            or metadata["row"]["matrix_hash"] != record.matrix_hash
            or int(config["reservoir_seed"]) != int(frame.set_index("csv_idx").loc[record.csv_idx, "seed"])
        ):
            raise ValueError(f"Training metadata identity mismatch: {key}")
        entries = dynamics[(dynamics.task == key[0]) & (dynamics.csv_idx == key[1]) & (dynamics.policy_seed == key[2])]
        if not entries.readin_hash.eq(record.readin_hash).all():
            raise ValueError(f"Training and dynamics read-in fingerprints disagree: {key}")
        if not args.smoke and (
            config["training_steps"] != 500000
            or not 500000 <= saved["num_timesteps"] < 500000 + saved["n_envs"] * saved["n_steps"]
        ):
            raise ValueError(f"Incomplete training budget: {key}")
        if not args.smoke and any(
            config.get(k) != v
            for k, v in dict(
                units=200, res_lr=0.1, res_sr=0.9, res_iss=1.0, del_obs=True, reset_res=True, skip_c=False
            ).items()
        ):
            raise ValueError(f"Encoder does not match corrected Solution 1 settings: {key}")
        for field in ("n_envs", "n_steps", "batch_size", "n_epochs"):
            if config[field] != saved[field]:
                raise ValueError(f"Checkpoint and configuration disagree on {field}: {key}")
        steps, values = read_scalars(log_path)
        if len(steps) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Missing/invalid training curves: {key}")
        curves[key] = (steps, values)
        source_hashes.update({str(model_path): digest, str(config_path): file_hash(config_path)})
        source_hashes.update({str(p): file_hash(p) for p in sorted(log_path.rglob("events.out.tfevents.*"))})
        audits.append(
            dict(
                task=key[0],
                csv_idx=key[1],
                policy_seed=key[2],
                model_sha256=digest,
                encoder_sha256=encoder_hash,
                matrix_hash=record.matrix_hash,
                readin_hash=record.readin_hash,
                n_envs=saved["n_envs"],
                n_steps=saved["n_steps"],
                batch_size=saved["batch_size"],
                n_epochs=saved["n_epochs"],
                rollout_samples=saved["n_envs"] * saved["n_steps"],
                timesteps=saved["num_timesteps"],
                optimizer_epochs=saved["_n_updates"],
                log_first=int(steps[0]),
                log_last=int(steps[-1]),
                log_points=len(steps),
                ppo_settings_sha256=hashlib.sha256(
                    json.dumps({k: config.get(k) for k in PPO_KEYS}, sort_keys=True).encode()
                ).hexdigest(),
            )
        )
        configurations.append(
            dict(
                task=key[0],
                csv_idx=key[1],
                policy_seed=key[2],
                environment=config,
                checkpoint_settings={
                    k: saved[k] for k in ("n_envs", "n_steps", "batch_size", "n_epochs", "num_timesteps")
                },
            )
        )
    payload_audit = []
    for record in paired.rename(columns={"return": "episode_return"}).itertuples():
        path = imported_artifact_path(record.path, args.trajectories_csv)
        with np.load(path, allow_pickle=False) as data:
            for field in (
                "task",
                "protocol",
                "csv_idx",
                "policy_seed",
                "episode",
                "evaluation_seed",
                "reservoir_seed",
                "matrix_hash",
                "readin_hash",
                "model_sha256",
                "encoder_sha256",
                "selection_sha256",
            ):
                if str(data[field]) != str(getattr(record, field)):
                    raise ValueError(f"Trajectory payload identity mismatch: {path} {field}")
            length = int(record.length)
            states, inputs = data["states"], data["inputs"]
            if states.shape != (length + 1, args.units) or len(inputs) != length + 1:
                raise ValueError(f"Trajectory terminal-state alignment mismatch: {path}")
            if not np.isfinite(states).all() or not np.isfinite(inputs).all():
                raise ValueError(f"Nonfinite trajectory: {path}")
            observations, actions, rewards = data["observations"], data["actions"], data["rewards"]
            if (
                not np.allclose(states[1:], data["contexts"])
                or not np.allclose(inputs[:-1, : observations.shape[1]], observations)
                or not np.allclose(inputs[:-1, observations.shape[1] : -1], actions)
                or not np.allclose(inputs[:-1, -1], rewards)
                or not np.isclose(rewards.sum(), record.episode_return)
            ):
                raise ValueError(f"Trajectory forcing/reward alignment mismatch: {path}")
        digest = file_hash(path)
        source_hashes[str(path)] = digest
        payload_audit.append(
            dict(
                task=record.task,
                csv_idx=record.csv_idx,
                policy_seed=record.policy_seed,
                episode=record.episode,
                evaluation_seed=record.evaluation_seed,
                length=length,
                path=str(path),
                sha256=digest,
            )
        )
    audit = pd.DataFrame(audits)
    settings = ["n_envs", "n_steps", "batch_size", "n_epochs", "ppo_settings_sha256"]
    regimes = audit.groupby(["task", *settings], as_index=False).size().to_dict("records")
    mixed_tasks = audit.groupby("task").apply(lambda x: len(x[settings].drop_duplicates()), include_groups=False)
    deviations = []
    if args.backend == "n4sid":
        deviations.append("SubspaceDMDc/N4SID follow-up departs from the original Solution 1 DMDc protocol")
    if args.exploratory_identification:
        deviations.append("Explicit exploratory identification/inference settings")
    if mixed_tasks.gt(1).any():
        deviations.append("Training settings differ within tasks: " + ", ".join(mixed_tasks[mixed_tasks.gt(1)].index))
    if args.smoke:
        deviations.append("Smoke validation: reduced cohort and analysis settings")
    if args.exploratory_reuse_ranking_episodes:
        deviations.append("Explicit exploratory analysis; ranking episodes may be reused for dynamics")
    audit.to_csv(output / "analysis" / "model_audit.csv", index=False)
    ranking.to_csv(output / "analysis" / "ranking_episodes.csv", index=False)
    dynamics_episodes.to_csv(output / "analysis" / "dynamics_episodes.csv", index=False)
    pd.DataFrame(payload_audit).to_csv(output / "analysis" / "trajectory_audit.csv", index=False)
    write_json(output / "analysis" / "model_settings.json", configurations)
    provenance = dict(
        identification_variant="subspace_n4sid" if args.backend == "n4sid" else "solution1_dmdc",
        source_hashes=source_hashes,
        training_regimes=regimes,
        training_run_history=training_history,
        deviations=deviations,
        interval_scope="Conditional on selected reservoirs and collected episode datasets; training-seed variability only",
        coverage="interim" if set(args.tasks) != set(TASKS) else "five tasks",
        confirmatory_eligible=not deviations and set(args.tasks) == set(TASKS),
    )
    write_json(output / "analysis" / "provenance.json", provenance)
    return ranking, dynamics, curves, provenance


def cached_distances(args, frame, task, output, dynamics):
    """Reuse only exact-identity fitted caches, validating source and output hashes."""
    settings = {
        k: getattr(args, k)
        for k in (
            "backend",
            "n_delays",
            "rank",
            "rank_energy",
            "min_rank",
            "max_rank",
            "dmd_regularization",
            "state_metric",
            "state_iters",
            "state_learning_rate",
            "analysis_seed",
            "device",
            "workers",
            "evaluation_episodes",
            "policy_seeds",
        )
    }
    relevant = dynamics[dynamics.task == task]
    payloads = {
        str((r.csv_idx, r.policy_seed, r.episode)): file_hash(imported_artifact_path(r.path, args.trajectories_csv))
        for r in relevant.itertuples()
    }
    sources = [ROOT / "src" / p for p in ("analysis.py", "utils.py", "dsa_episodes.py", "dsa_numerics.py")]
    sources.extend(sorted((ROOT / "DSA" / "DSA").rglob("*.py")))
    identity = dict(
        schema=4,
        settings=settings,
        identification=dict(
            adapter="EpisodeSeparatedSubspaceDMDc" if args.backend == "n4sid" else "EpisodeSeparatedDMDc",
            dtype="float64" if args.backend == "n4sid" else "backend_native",
            rank_selection="explicit" if args.rank is not None else "delay_embedded_state_energy",
        ),
        selection_sha256=file_hash(args.selected_csv),
        reservoir_ids=frame.csv_idx.tolist(),
        payloads=payloads,
        fit_sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "dsa-metric")},
    )
    names = [
        f"{task}_{suffix}"
        for suffix in ("policy_distances.npz", "policy_order.csv", "reservoir_distances.npz", "diagnostics.json")
    ]
    cache_name = f"{task}_distance_cache.json"
    if args.distance_cache:
        cache = Path(args.distance_cache).resolve()
        cache = cache / "analysis" if (cache / "analysis").is_dir() else cache
        metadata = json.loads((cache / cache_name).read_text())
        if metadata["identity"] != identity:
            raise ValueError(f"Distance cache input/settings/source mismatch: {task}; refit without --distance-cache")
        for name in names:
            if file_hash(cache / name) != metadata["files"][name]:
                raise ValueError(f"Distance cache output hash mismatch: {name}")
            shutil.copyfile(cache / name, output / "analysis" / name)
        shutil.copyfile(cache / cache_name, output / "analysis" / cache_name)
    else:
        cohort, _, diagnostics, excluded = fit_policy_trajectories(args, frame, task, output)
        if (
            excluded
            or cohort.csv_idx.tolist() != frame.csv_idx.tolist()
            or not diagnostics["episode_boundaries_preserved"]
        ):
            raise ValueError("Incomplete or boundary-unsafe fitted cohort")
        write_json(output / "analysis" / names[-1], diagnostics)
        write_json(
            output / "analysis" / cache_name,
            dict(identity=identity, files={n: file_hash(output / "analysis" / n) for n in names}),
        )
    diagnostics = json.loads((output / "analysis" / names[-1]).read_text())
    order = pd.read_csv(output / "analysis" / names[1], keep_default_na=False)
    with np.load(output / "analysis" / names[0], allow_pickle=False) as data:
        matrices = {name: data[name] for name in METRICS}
        if not np.array_equal(data["csv_idx"], order.csv_idx) or not np.array_equal(
            data["policy_seed"], order.policy_seed
        ):
            raise ValueError("Cache system ordering mismatch")
    if (
        not diagnostics["episode_boundaries_preserved"]
        or diagnostics.get("dynamics_schema_version") != 3
        or diagnostics.get("backend") != args.backend
        or diagnostics["fitted_system_count"] != len(frame) * len(args.policy_seeds)
        or diagnostics["system_order"] != order[["csv_idx", "policy_seed", "episode_count"]].to_dict("records")
    ):
        raise ValueError("Cache diagnostics do not describe the complete policy-specific cohort")
    aggregated = aggregate_policy_distances(matrices, order, frame.csv_idx, args.policy_seeds)
    with np.load(output / "analysis" / names[2], allow_pickle=False) as saved:
        if not np.array_equal(saved["csv_idx"], frame.csv_idx):
            raise ValueError("Cached reservoir order mismatch")
        for name in METRICS:
            if not np.allclose(saved[name], aggregated[name]):
                raise ValueError("Cached aggregation does not equal all cross-policy means")
    return matrices, aggregated, order, diagnostics
