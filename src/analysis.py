"""InputDSA orchestration and reservoir-level hypothesis tests.

H1 compares top/bottom within-group distances. H2 compares descriptor distance
with absolute return difference. Permutations operate on reservoir labels.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

from src.reservoirs import load_reservoir, reservoir_path
from src.utils import (
    choose_rank,
    delay_embed,
    experiment_name,
    file_hash,
    legacy_config,
    read_scalars,
    standardize,
    weighted_performance,
    write_json,
)

METRICS = ("joint", "state_joint", "control_joint", "state_separate", "control_separate")


def load_dsa():
    """Import the external local DSA package without modifying its source."""
    import sys

    from src.utils import ROOT

    # The checkout is external and can be used before editable installation.
    checkout = str(ROOT / "DSA")
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    from DSA import DMDc, InputDSA, SubspaceDMDc

    return DMDc, InputDSA, SubspaceDMDc


def fit_distances(states, inputs, args):
    """Fit five InputDSA matrices, preserving independent trial boundaries.

    Corrected dmdc uses DSA.DMDc. Legacy dmdc preserves the historical string
    alias routed by DSA.SubspaceDMDc to its custom subspace implementation.
    """
    import torch

    if len(states) < 2 or len(states) != len(inputs):
        raise ValueError("Need at least two aligned systems")
    for xs, us in zip(states, inputs):
        xlist = [xs] if isinstance(xs, np.ndarray) else xs
        ulist = [us] if isinstance(us, np.ndarray) else us
        if len(xlist) != len(ulist):
            raise ValueError("Episode count mismatch")
        for x, u in zip(xlist, ulist):
            delay_embed(x, args.n_delays)
            delay_embed(u, args.n_delays)
            if len(x) != len(u):
                raise ValueError("State/input time alignment mismatch")
    automatic, ranks = choose_rank(states, args.n_delays, args.rank_energy, args.min_rank, args.max_rank)
    rank = automatic if args.rank is None else args.rank
    support = min(np.linalg.matrix_rank(delay_embed(x, args.n_delays)) for x in states)
    if rank < 1 or rank > support:
        raise ValueError(f"Requested rank {rank} exceeds common support {support}")
    torch.manual_seed(args.analysis_seed)
    DMDc, InputDSA, SubspaceDMDc = load_dsa()
    from src.dsa_episodes import EpisodeSeparatedDMDc
    from src.dsa_numerics import StableControllabilityDistance

    if args.protocol == "corrected" and args.backend == "dmdc":
        klass = DMDc
        config = {"n_delays": args.n_delays, "rank_output": rank, "rank_input": None, "lamb": args.dmd_regularization}
    else:
        klass = SubspaceDMDc
        config = {"n_delays": args.n_delays, "rank": rank, "backend": args.backend, "lamb": args.dmd_regularization}
    engine = InputDSA(
        X=states,
        X_control=inputs,
        dmd_class=klass,
        dmd_config=config,
        simdist_config={"compare": "joint", "return_distance_components": True},
        n_jobs=args.workers,
        device=args.device,
    )
    engine.simdist = StableControllabilityDistance(**engine.simdist_config)
    if args.protocol == "corrected":
        if args.backend != "dmdc":
            raise ValueError("Corrected episode-separated dynamics require --backend dmdc")
        # InputDSA recognizes exact external classes only. Register DMDc through
        # its public constructor, then supply adapters as its fitted systems.
        engine.dmds = [
            [
                EpisodeSeparatedDMDc(
                    data=xs,
                    control_data=us,
                    n_delays=args.n_delays,
                    rank_output=rank,
                    rank_input=None,
                    lamb=args.dmd_regularization,
                    device=args.device,
                )
                for xs, us in zip(states, inputs)
            ]
        ]
    joint = engine.fit_score()
    result = {"joint": joint[:, :, 0], "state_joint": joint[:, :, 1], "control_joint": joint[:, :, 2]}
    engine.update_compare_method(
        compare="state",
        simdist_config={"score_method": args.state_metric, "iters": args.state_iters, "lr": args.state_learning_rate},
    )
    result["state_separate"] = engine.score()
    engine.update_compare_method(compare="control", simdist_config={"score_method": "euclidean"})
    engine.simdist = StableControllabilityDistance(**engine.simdist_config)
    result["control_separate"] = engine.score()
    diagnostics = []
    for model in engine.dmds[0]:

        def array(value):
            """Convert an external operator to a NumPy diagnostic array."""
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        a, b = array(model.A_v), array(model.B_v)
        diagnostics.append(
            {
                "state_operator_norm": float(np.linalg.norm(a)),
                "state_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a.astype(np.float64))))),
                "control_operator_norm": float(np.linalg.norm(b)),
                "state_condition": float(np.linalg.cond(a)),
                "control_singular_values": np.linalg.svd(b, compute_uv=False).tolist(),
                "episode_transition_counts": getattr(model, "episode_transition_counts", None),
            }
        )
    for name, matrix in result.items():
        if matrix.shape != (len(states), len(states)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {name} distances; backend={args.backend}; no fallback applied")
    return result, {
        "rank": rank,
        "per_system_energy_ranks": ranks,
        "n_delays": args.n_delays,
        "backend": args.backend,
        "dmd_class": klass.__name__,
        "episode_boundaries_preserved": args.protocol == "corrected",
        "dmd_adapter": "EpisodeSeparatedDMDc" if args.protocol == "corrected" else None,
        "controllability_precision": "float64",
        "controllability_alignment": "scaled_cross_product",
        "operators": diagnostics,
        "rank_capped": max(ranks) > rank,
    }


def get_performance(task_prefix, window_steps=10000, weight_type="linear", logs_dir=None, models_dir=None):
    """Rank historical experiment folders by all available last-window events."""
    from src.utils import ROOT

    logs_dir = Path(logs_dir) if logs_dir is not None else ROOT / "rl_only/logs_all"
    models_dir = Path(models_dir) if models_dir is not None else ROOT / "rl_only/models_all"
    results = []
    for directory in sorted(logs_dir.glob(f"{task_prefix}_*")):
        if not directory.is_dir():
            continue
        steps, values = read_scalars(directory)
        score = weighted_performance(steps, values, window_steps, weight_type)
        models = sorted((models_dir / directory.name).glob("*.zip"))
        results.append(
            {
                "seed_name": directory.name,
                "performance": score,
                "log_path": str(directory),
                "model_path": str(models[0]) if models else "Model not found",
            }
        )
    return sorted(results, key=lambda row: (-row["performance"], row["seed_name"]))


def performance_table(args, frame):
    """Return reservoir mean evaluation returns or explicitly requested log rankings."""
    if args.ranking_source == "evaluation":
        if args.evaluation_csv is None:
            raise ValueError("Evaluation ranking and H2 require --evaluation-csv episodes.csv")
        episodes = pd.read_csv(args.evaluation_csv)
        required = {"task", "csv_idx", "policy_seed", "return", "protocol"}
        if required - set(episodes) or not episodes.protocol.eq(args.protocol).all():
            raise ValueError("Evaluation table columns or protocol mismatch")
        manifest = Path(args.evaluation_csv).parent.parent / "manifest.json"
        if not manifest.exists():
            raise ValueError("Evaluation table must have its originating run manifest")
        import json

        metadata = json.loads(manifest.read_text())
        if metadata.get("selection_sha256") != file_hash(args.selected_csv):
            raise ValueError("Evaluation selection fingerprint mismatch")
        episodes = episodes[(episodes.csv_idx >= 0) & episodes.task.isin(args.tasks)]
        if args.protocol == "corrected":
            episodes = episodes[episodes.policy_seed.isin(args.policy_seeds)]
        per_seed = episodes.groupby(["task", "csv_idx", "policy_seed"], as_index=False)["return"].mean()
        result = (
            per_seed.groupby(["task", "csv_idx"], as_index=False)["return"].agg(["mean", "std", "count"]).reset_index()
        )
        return result.rename(columns={"mean": "performance", "std": "policy_std", "count": "policy_replicates"})
    records = []
    for task in args.tasks:
        rankings = {
            item["seed_name"]: item
            for item in get_performance(task, args.window_steps, args.weight_type, args.logs_dir, args.models_dir)
        }
        for _, row in frame.iterrows():
            name, _ = experiment_name(task, row.stratum, args.training_steps, int(row.csv_idx))
            if name in rankings and np.isfinite(rankings[name]["performance"]):
                records.append(
                    {
                        "task": task,
                        "csv_idx": int(row.csv_idx),
                        "performance": rankings[name]["performance"],
                        "policy_replicates": 1,
                    }
                )
    return pd.DataFrame(records, columns=["task", "csv_idx", "performance", "policy_replicates"])


def rank_groups(frame, scores, group_size):
    """Choose nonoverlapping top/bottom reservoirs; tie-break by stable csv_idx."""
    merged = frame.merge(scores, on="csv_idx", validate="one_to_one")
    merged = merged[np.isfinite(merged.performance)].sort_values(["performance", "csv_idx"], ascending=[False, True])
    if len(merged) < 2 * group_size or group_size < 2:
        raise ValueError("Insufficient reservoirs for two nonoverlapping groups of at least two")
    top, bottom = merged.head(group_size).copy(), merged.tail(group_size).copy()
    top["group"], bottom["group"] = "Top", "Bottom"
    return pd.concat([top.sort_values("stratum", kind="stable"), bottom.sort_values("stratum", kind="stable")])


def load_trajectories(args, frame, task, policy_seed=None):
    """Load aligned system trials, returning filtered metadata and exclusions."""
    registry = pd.read_csv(args.trajectories_csv, keep_default_na=False) if args.trajectories_csv else None
    if args.protocol == "corrected" and registry is None:
        raise ValueError("Corrected analyses require --trajectories-csv")
    if registry is not None and not registry.protocol.eq(args.protocol).all():
        raise ValueError("Trajectory registry protocol mismatch")
    if (
        registry is not None
        and "selection_sha256" in registry
        and not registry.selection_sha256.eq(file_hash(args.selected_csv)).all()
    ):
        raise ValueError("Trajectory selection fingerprint mismatch")
    states, inputs, rows, excluded = [], [], [], []
    for _, row in frame.iterrows():
        try:
            if registry is None:
                prefix = Path(args.trajectories_dir) / f"{task}_{row.stratum}_{int(row.seed)}"
                x = np.load(str(prefix) + "_Ct.npy", allow_pickle=False)
                u = np.load(str(prefix) + "_Ot.npy", allow_pickle=False)
                xs, us = x, u
            else:
                entries = registry[(registry.task == task) & (registry.csv_idx == int(row.csv_idx))].sort_values(
                    ["policy_seed", "episode"]
                )
                if policy_seed is not None:
                    entries = entries[entries.policy_seed == policy_seed]
                if entries.empty:
                    raise FileNotFoundError("No registered episodes")
                if policy_seed is not None:
                    expected = set(range(args.evaluation_episodes))
                    if entries.episode.duplicated().any() or set(entries.episode) != expected:
                        raise ValueError(
                            f"Incomplete/duplicate episodes for {task} idx {row.csv_idx} policy {policy_seed}"
                        )
                xs, us = [], []
                for _, entry in entries.iterrows():
                    if int(entry.reservoir_seed) != int(row.seed):
                        raise ValueError("Trajectory reservoir seed mismatch")
                    if args.protocol == "corrected" and entry.matrix_hash != row.matrix_hash:
                        raise ValueError("Trajectory matrix hash mismatch")
                    with np.load(entry.path, allow_pickle=False) as data:
                        if str(data["protocol"]) != args.protocol or int(data["csv_idx"]) != int(row.csv_idx):
                            raise ValueError("Trajectory payload identity mismatch")
                        if policy_seed is not None:
                            if (
                                int(data["policy_seed"]) != policy_seed
                                or int(data["episode"]) != int(entry.episode)
                                or int(data["reservoir_seed"]) != int(row.seed)
                            ):
                                raise ValueError("Trajectory policy/episode payload identity mismatch")
                            if (
                                str(data["matrix_hash"]) != entry.matrix_hash
                                or str(data["readin_hash"]) != entry.readin_hash
                            ):
                                raise ValueError("Trajectory payload matrix fingerprint mismatch")
                        xs.append(data["states"])
                        us.append(data["inputs"])
            states.append(xs)
            inputs.append(us)
            rows.append(row.to_dict())
        except FileNotFoundError as error:
            if not args.allow_subset:
                raise FileNotFoundError(f"Missing {task} csv_idx={row.csv_idx}: {error}") from error
            excluded.append({"task": task, "csv_idx": int(row.csv_idx), "reason": str(error)})
    if len(rows) < 2:
        raise ValueError("Fewer than two complete trajectories remain")
    return pd.DataFrame(rows), states, inputs, excluded


def fit_policy_trajectories(args, frame, task, output):
    """Fit one system per reservoir/policy and average all cross-policy pairs."""
    if args.protocol != "corrected" or args.backend != "dmdc":
        raise ValueError("Policy-specific dynamics require corrected DMDc")
    registry = pd.read_csv(args.trajectories_csv, keep_default_na=False)
    relevant = registry[
        (registry.task == task) & registry.csv_idx.isin(frame.csv_idx) & registry.policy_seed.isin(args.policy_seeds)
    ]
    if (
        "readin_hash" not in relevant
        or relevant.readin_hash.eq("").any()
        or relevant.groupby("csv_idx").readin_hash.nunique().ne(1).any()
    ):
        raise ValueError("Policy replicates must share the same read-in matrix per reservoir/task")
    systems, xs, us, exclusions = [], [], [], []
    complete = set(frame.csv_idx)
    for seed in args.policy_seeds:
        rows, states, inputs, excluded = load_trajectories(args, frame, task, policy_seed=seed)
        complete.intersection_update(rows.csv_idx)
        exclusions.extend({**entry, "policy_seed": seed} for entry in excluded)
        for (_, row), x, u in zip(rows.iterrows(), states, inputs):
            systems.append({**row.to_dict(), "policy_seed": seed, "episode_count": len(x)})
            xs.append(x)
            us.append(u)
    positions = [i for i, row in enumerate(systems) if row["csv_idx"] in complete]
    systems = pd.DataFrame([systems[i] for i in positions])
    if len(complete) < 2:
        raise ValueError("Need at least two reservoirs with all requested policy replicates")
    xs, us = [xs[i] for i in positions], [us[i] for i in positions]
    matrices, diagnostics = fit_distances(xs, us, args)
    cohort = frame[frame.csv_idx.isin(complete)].copy()
    aggregated = aggregate_policy_distances(matrices, systems, cohort.csv_idx, args.policy_seeds)
    diagnostics.update(
        dynamics_schema_version=2,
        aggregation="mean_all_cross_policy_pairs",
        policy_seeds=args.policy_seeds,
        fitted_system_count=len(systems),
        pairs_per_reservoir_pair=len(args.policy_seeds) ** 2,
        system_order=systems[["csv_idx", "policy_seed", "episode_count"]].to_dict("records"),
    )
    np.savez_compressed(
        output / "analysis" / f"{task}_reservoir_distances.npz",
        **aggregated,
        csv_idx=cohort.csv_idx.to_numpy(),
        policy_seeds=np.asarray(args.policy_seeds),
        rank=diagnostics["rank"],
        n_delays=args.n_delays,
        protocol=args.protocol,
    )
    np.savez_compressed(
        output / "analysis" / f"{task}_policy_distances.npz",
        **matrices,
        csv_idx=systems.csv_idx.to_numpy(),
        policy_seed=systems.policy_seed.to_numpy(),
        rank=diagnostics["rank"],
        n_delays=args.n_delays,
        protocol=args.protocol,
    )
    systems.to_csv(output / "analysis" / f"{task}_policy_order.csv", index=False)
    return cohort, aggregated, diagnostics, exclusions


def aggregate_policy_distances(matrices, systems, reservoir_ids, policy_seeds):
    """Average the complete Cartesian product of policy pairs for distinct reservoirs."""
    blocks = []
    for index in reservoir_ids:
        rows = systems[systems.csv_idx == index]
        if rows.policy_seed.duplicated().any() or set(rows.policy_seed) != set(policy_seeds):
            raise ValueError(f"Incomplete policy systems for reservoir {index}")
        blocks.append(rows.index.to_numpy())
    result = {}
    for name, matrix in matrices.items():
        if matrix.shape != (len(systems), len(systems)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid policy distance matrix: {name}")
        if not np.allclose(matrix, matrix.T):
            raise ValueError(f"Asymmetric policy distance matrix: {name}")
        distance = np.zeros((len(blocks), len(blocks)))
        for i, a in enumerate(blocks):
            for j in range(i):
                distance[i, j] = distance[j, i] = matrix[np.ix_(a, blocks[j])].mean()
        result[name] = distance
    return result


def ou_noise(n_timesteps, dim=10, theta=0.15, seed=0):
    """Return standardized (time, features) discrete Ornstein–Uhlenbeck inputs."""
    if n_timesteps < 2 or dim < 1 or not 0 < theta < 2:
        raise ValueError("Require >=2 samples, positive dimension, and 0 < theta < 2")
    rng = np.random.default_rng(seed)
    state = np.zeros(dim)
    output = np.empty((n_timesteps, dim))
    for index in range(n_timesteps):
        state = state * (1 - theta) + np.sqrt(2 * theta) * rng.standard_normal(dim)
        output[index] = state
    return standardize(output)


def synthetic_trajectories(args, frame, condition):
    """Run common seeded synthetic input through each fixed reservoir."""
    from reservoirpy.datasets import mackey_glass
    from reservoirpy.mat_gen import normal
    from reservoirpy.nodes import Reservoir
    from reservoirpy.nodes.wsbm_esn import AssortativeESN, CorePeripheryESN, DisassortativeESN, MixedESN

    control = (
        mackey_glass(n_timesteps=args.input_steps, seed=args.input_seed).reshape(-1, 1)
        if condition == "mackey_glass"
        else ou_noise(args.input_steps, args.input_dim, args.ou_theta, args.input_seed)
    )
    states, inputs = [], []
    for _, row in frame.iterrows():
        if args.protocol == "corrected":
            artifact = load_reservoir(reservoir_path(args.selected_csv, row), row, args.units, args.res_sr)
            node = Reservoir(
                units=args.units,
                W=artifact["W"].copy(),
                seed=int(row.seed),
                sr=args.res_sr,
                lr=args.res_lr,
                input_scaling=args.res_iss,
                Win=normal,
                input_connectivity=1.0,
            )
        else:
            config = legacy_config(row)
            motif = config["motif"]
            common = {key: config[key] for key in ("n_communities", "connectivity", "sigma", "p_negative", "seed")}
            if motif in {"assortative", "null"}:
                node = AssortativeESN(
                    mu_in=config["mid"] if motif == "null" else config["hi"],
                    mu_out=config["mid"] if motif == "null" else config["lo"],
                    units=args.units,
                    **common,
                )
            elif motif == "disassortative":
                node = DisassortativeESN(mu_in=config["hi"], mu_out=config["lo"], units=args.units, **common)
            elif motif == "core_periphery":
                node = CorePeripheryESN(
                    n_reservoir=args.units,
                    mu_cc=config["hi"],
                    mu_cp=config["mid"],
                    mu_pp=config["lo"],
                    core_fraction=config["core_fraction"],
                    **common,
                )
            else:
                node = MixedESN(hi=config["hi"], lo=config["lo"], units=args.units, **common)
        state = node.run(control)
        if args.protocol == "corrected":
            # x[t] is post-input; x[t+1] is driven by the next input.
            states.append(state[:-1])
            inputs.append(control[1:])
        else:
            states.append(state)
            inputs.append(control)
    return states, inputs


def rank_correlation(a, b):
    """Return Spearman rho, or NaN for an undefined constant-distance test."""
    x, y = rankdata(a), rankdata(b)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def h2_test(descriptors, returns, permutations=9999, seed=0):
    """Test distance/return association by permuting whole reservoir labels."""
    if len(returns) < 3 or permutations < 1:
        raise ValueError("H2 requires >=3 reservoirs and positive permutations")
    structural = pdist(descriptors)
    behavioral = pdist(np.asarray(returns)[:, None], metric="cityblock")
    observed = rank_correlation(structural, behavioral)
    if not np.isfinite(observed):
        return {"rho": observed, "p": float("nan"), "n_reservoirs": len(returns), "n_pairs": len(structural)}
    rng, extreme = np.random.default_rng(seed), 0
    for _ in range(permutations):
        permuted = pdist(rng.permutation(returns)[:, None], metric="cityblock")
        extreme += abs(rank_correlation(structural, permuted)) >= abs(observed) - 1e-12
    return {
        "rho": observed,
        "p": (extreme + 1) / (permutations + 1),
        "n_reservoirs": len(returns),
        "n_pairs": len(structural),
    }


def h1_test(matrix, groups, permutations=9999, seed=0):
    """Compare bottom-minus-top mean distance using group-label permutations."""
    groups = np.asarray(groups)

    def mean(indices):
        """Mean strict-upper-triangle distance for a group of reservoirs."""
        block = matrix[np.ix_(indices, indices)]
        return float(block[np.triu_indices(len(indices), 1)].mean())

    top, bottom = np.flatnonzero(groups == "Top"), np.flatnonzero(groups == "Bottom")
    if len(top) < 2 or len(bottom) < 2:
        raise ValueError("H1 requires >=2 reservoirs in each group")
    observed = mean(bottom) - mean(top)
    rng, extreme = np.random.default_rng(seed), 0
    for _ in range(permutations):
        order = rng.permutation(len(groups))
        extreme += mean(order[len(top) :]) - mean(order[: len(top)]) >= observed - 1e-12
    return {
        "top_mean": mean(top),
        "bottom_mean": mean(bottom),
        "difference": observed,
        "p": (extreme + 1) / (permutations + 1),
        "n_top": len(top),
        "n_bottom": len(bottom),
    }


def holm(values):
    """Return Holm-adjusted p-values, preserving undefined tests as NaN."""
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    adjusted = np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1))
    output[order] = np.minimum(adjusted, 1)
    return output


def save_distances(output, task, frame, matrices, diagnostics, args):
    """Export ordered matrices, metadata, and a headless five-panel figure."""
    from src.figures import plot_distances

    identifier = re.sub(r"[^A-Za-z0-9_-]", "_", task)
    labels = [f"{row.get('group', '')}_{row['stratum']}_{row['csv_idx']}" for _, row in frame.iterrows()]
    np.savez_compressed(
        output / "analysis" / f"{identifier}_distances.npz",
        **matrices,
        labels=np.asarray(labels),
        csv_idx=frame.csv_idx.to_numpy(),
        groups=frame["group"].to_numpy(dtype=str) if "group" in frame else np.array([], dtype=str),
        rank=diagnostics["rank"],
        n_delays=args.n_delays,
        protocol=args.protocol,
    )
    frame.to_csv(output / "analysis" / f"{identifier}_order.csv", index=False)
    write_json(output / "analysis" / f"{identifier}_diagnostics.json", diagnostics)
    plot_distances(matrices, labels, output / "figures" / f"{identifier}_distances.png", args.dpi)
