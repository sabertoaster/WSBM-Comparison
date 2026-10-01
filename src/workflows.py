"""Composable research workflows behind all supported script entry points."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.utils import DESCRIPTORS, STRATA, read_selection, standardize, write_json


def describe_candidate(payload):
    """Describe a structure and its checksum without retaining every candidate W."""
    from src.reservoirs import build_corrected
    from src.selection import build, descriptors

    row, config = payload
    try:
        if config["protocol"] == "corrected":
            artifact = build_corrected(row, config["units"], config["sigma_base"], config["res_sr"])
            values = descriptors(artifact["W"], artifact["assignments"], weighted=True)
            row = {**row, "matrix_hash": str(artifact["matrix_hash"])}
        else:
            node = build(row, config["units"], config["sigma_base"], config["res_sr"])
            node.initialize(np.zeros((1, 1)))
            values = descriptors(node.W, node.community_assignments)
            artifact = None
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite graph descriptors")
        return {**row, **values}, None, None
    except (ValueError, np.linalg.LinAlgError) as error:
        return row, None, str(error)


def select(args, output):
    """Sample, standardize, and MaxMin-select structures; persist chosen matrices."""
    from src.figures import plot_selection
    from src.reservoirs import build_corrected, save_reservoir
    from src.selection import maxmin, sample_params

    rng = np.random.default_rng(args.selection_seed)
    strata = args.strata or STRATA
    parameters = [
        sample_params(
            stratum,
            rng,
            **{
                key: getattr(args, key)
                for key in (
                    "communities",
                    "density_range",
                    "centre_range",
                    "dispersion_range",
                    "topology_range",
                    "size_alphas",
                    "contrast_range",
                    "core_range",
                    "mixing_range",
                    "null_centre_range",
                    "core_contrast_range",
                    "mixed_contrast_range",
                )
            },
        )
        for stratum in strata
        for _ in range(args.pool_per_stratum)
    ]
    config = {key: getattr(args, key) for key in ("protocol", "units", "sigma_base", "res_sr")}
    payloads = [(row, config) for row in parameters]
    rows, rejected = [], []
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = executor.map(describe_candidate, payloads) if executor else map(describe_candidate, payloads)
        for row, _, reason in results:
            if reason:
                rejected.append({**row, "reason": reason})
            else:
                rows.append(row)
    finally:
        if executor:
            executor.shutdown()
    pd.DataFrame(rejected).to_csv(output / "selection" / "rejections.csv", index=False)
    if not rows:
        raise ValueError("All candidate reservoirs rejected; see selection/rejections.csv")
    pool = pd.DataFrame(rows)
    pool["units"] = args.units
    pool["protocol"] = args.protocol
    values = pool[list(DESCRIPTORS)].to_numpy()
    z = standardize(values)
    u, singular, rotation = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    if np.sum(singular**2) == 0:
        raise ValueError("Structural pool has no descriptor variation")
    pcs, explained = u * singular, singular**2 / np.sum(singular**2)
    picked = []
    for stratum in strata:
        indices = np.flatnonzero(pool.stratum.to_numpy() == stratum)
        if len(indices) < args.select_per_stratum:
            raise ValueError(
                f"Only {len(indices)} valid {stratum} candidates for {args.select_per_stratum} picks; increase pool"
            )
        choices = maxmin(pcs[indices, : args.pca_components], args.select_per_stratum, rng)
        if len(set(choices)) != len(choices):
            raise ValueError("Selection produced duplicate candidates")
        picked.extend(indices[choices])
    selected = pool.iloc[picked].copy().reset_index(drop=True)
    selected.insert(0, "csv_idx", np.arange(len(selected)))
    selected["pool_idx"] = picked
    selected["units"] = args.units
    selected["protocol"] = args.protocol
    if args.protocol == "corrected":
        matrix_directory = output / "selection" / "matrices"
        matrix_directory.mkdir(exist_ok=True)
        files, hashes = [], []
        for index, pool_index in enumerate(picked):
            artifact = build_corrected(pool.iloc[pool_index], args.units, args.sigma_base, args.res_sr)
            if str(artifact["matrix_hash"]) != pool.iloc[pool_index]["matrix_hash"]:
                raise ValueError("Selected matrix differs from the matrix used for descriptors")
            filename = f"matrices/reservoir_{index}.npz"
            save_reservoir(output / "selection" / filename, artifact)
            files.append(filename)
            hashes.append(str(artifact["matrix_hash"]))
        selected["matrix_file"], selected["matrix_hash"] = files, hashes
    pool.to_csv(output / "selection" / "descriptors.csv", index=False)
    selected.to_csv(output / "selection" / "selected.csv", index=False)
    np.savez_compressed(
        output / "selection" / "pca.npz",
        pcs=pcs,
        mean=values.mean(axis=0),
        scale=values.std(axis=0),
        rotation=rotation,
        explained=explained,
    )
    plot_selection(pool, picked, pcs, explained, output / "figures" / "selection.png", args.dpi)
    print(f"Selected {len(selected)} configurations, {args.units} neurons each; rejected {len(rejected)} draws")
    return {
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected_csv": str(output / "selection" / "selected.csv"),
    }


def pool_reference(args, required=False):
    """Load the explicit full pool, or its corrected selection sibling."""
    from pathlib import Path

    path = args.pool_csv
    if path is None:
        if args.protocol == "corrected":
            path = Path(args.selected_csv).parent / "descriptors.csv"
        elif required:
            from src.utils import ROOT

            path = ROOT / "scripts/selection/descriptors.csv"
    if path is None:
        return None
    return read_selection(path)


def analyze_distances(args, frame, output, workflow):
    """Analyze real or synthetic trajectories with one stable ordering per export."""
    from src.analysis import (
        METRICS,
        fit_distances,
        h1_test,
        holm,
        load_trajectories,
        performance_table,
        rank_groups,
        save_distances,
        synthetic_trajectories,
    )
    from src.config import SYNTHETIC
    from src.figures import plot_structural

    if args.limit:
        frame = frame.head(args.limit)
    if workflow == "inputdsa_5_systems_mackey_glass_ou":
        frame = frame.drop_duplicates("stratum")
    top_bottom = workflow in {"dev_scripts", "inputdsa_top_bottom_10_reservoirs_H1", "analyze_hypotheses"}
    scores = performance_table(args, frame) if top_bottom else None
    exclusions, tests = [], []
    identifiers = args.conditions if workflow in SYNTHETIC else args.tasks
    for identifier in identifiers:
        selected = (
            frame.sort_values("stratum", kind="stable")
            if workflow in SYNTHETIC | {"inputdsa_100_reservoirs_H2"}
            else frame.copy()
        )
        if top_bottom:
            selected = rank_groups(frame, scores[scores.task == identifier].drop(columns="task"), args.group_size)
        if workflow in SYNTHETIC:
            xs, us = synthetic_trajectories(args, selected, identifier)
        else:
            selected, xs, us, excluded = load_trajectories(args, selected, identifier)
            exclusions.extend(excluded)
        matrices, diagnostics = fit_distances(xs, us, args)
        save_distances(output, identifier, selected, matrices, diagnostics, args)
        if workflow in {"inputdsa_100_reservoirs_H1", "inputdsa_top_bottom_10_reservoirs_H1"}:
            pool = pool_reference(args) if args.protocol == "corrected" else None
            reference = pool[list(DESCRIPTORS)].to_numpy() if pool is not None else None
            pairs = plot_structural(
                selected,
                matrices["state_joint"],
                output / "figures" / f"{identifier}_structural_state.png",
                reference,
                args.dpi,
            )
            pairs.to_csv(output / "analysis" / f"{identifier}_pairs.csv", index=False)
        if workflow in {"dev_scripts", "analyze_hypotheses"}:
            for name in METRICS:
                tests.append(
                    {
                        "task": identifier,
                        "metric": name,
                        **h1_test(matrices[name], selected.group, args.permutations, args.analysis_seed),
                    }
                )
    if tests:
        table = pd.DataFrame(tests)
        table["p_holm"] = holm(table.p)
        table.to_csv(output / "analysis" / "h1.csv", index=False)
    write_json(output / "analysis" / "exclusions.json", exclusions)
    return {"h1_tests": len(tests), "excluded_count": len(exclusions)}


def analyze_h2(args, frame, output):
    """Export global and within-stratum reservoir-level H2 permutation tests."""
    from scipy.spatial.distance import pdist

    from src.analysis import h2_test, holm, performance_table
    from src.figures import pyplot

    if args.ranking_source != "evaluation":
        raise ValueError("Paper H2 requires deterministic returns; use --ranking-source evaluation")
    scores = performance_table(args, frame)
    pool = pool_reference(args, required=args.protocol == "corrected")
    reference = pool[list(DESCRIPTORS)].to_numpy() if pool is not None else None
    records, exclusions, behavior = [], [], []
    for task in args.tasks:
        values = scores[scores.task == task].drop(columns="task")
        missing = frame.loc[~frame.csv_idx.isin(values.csv_idx), "csv_idx"].tolist()
        if missing and not args.allow_subset:
            raise ValueError(f"Missing {task} evaluation returns for csv_idx {missing}")
        exclusions.extend({"task": task, "csv_idx": index, "reason": "missing evaluation returns"} for index in missing)
        merged = frame.merge(values, on="csv_idx", validate="one_to_one")
        merged["performance"] = pd.to_numeric(merged.performance)
        if not np.isfinite(merged.performance).all():
            raise ValueError("Nonfinite evaluation returns")
        if merged.empty:
            raise ValueError(f"No {task} reservoir returns remain")
        returns = merged.performance.to_numpy()
        behavior.append(
            {
                "task": task,
                "n_reservoirs": len(merged),
                "return_min": float(returns.min()),
                "return_max": float(returns.max()),
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "coefficient_of_variation": float(returns.std() / abs(returns.mean())) if returns.mean() else np.nan,
                "min_policy_replicates": int(merged.policy_replicates.min()),
                "max_policy_replicates": int(merged.policy_replicates.max()),
            }
        )
        population = reference if reference is not None else merged[list(DESCRIPTORS)].to_numpy()
        for stratum in ["all", *STRATA]:
            subset = merged if stratum == "all" else merged[merged.stratum == stratum]
            if len(subset) < 3:
                records.append(
                    {
                        "task": task,
                        "stratum": stratum,
                        "rho": np.nan,
                        "p": np.nan,
                        "n_reservoirs": len(subset),
                        "n_pairs": len(subset) * (len(subset) - 1) // 2,
                        "status": "insufficient reservoirs",
                    }
                )
                continue
            z = standardize(subset[list(DESCRIPTORS)].to_numpy(), population)
            test = h2_test(z, subset.performance.to_numpy(), args.permutations, args.analysis_seed)
            records.append(
                {
                    "task": task,
                    "stratum": stratum,
                    **test,
                    "status": "ok" if np.isfinite(test["rho"]) else "constant distances",
                }
            )
            if stratum == "all":
                plt = pyplot()
                structural, returns = pdist(z), pdist(subset.performance.to_numpy()[:, None], metric="cityblock")
                fig, axis = plt.subplots(figsize=(6, 5))
                axis.scatter(structural, returns, s=6, alpha=0.2)
                axis.set_xlabel("Structural distance")
                axis.set_ylabel("Absolute mean-return difference")
                fig.savefig(output / "figures" / f"{task}_h2.png", dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                indices = np.triu_indices(len(subset), 1)
                ids = subset.csv_idx.to_numpy()
                pd.DataFrame(
                    {
                        "csv_idx_a": ids[indices[0]],
                        "csv_idx_b": ids[indices[1]],
                        "structural_distance": structural,
                        "absolute_return_difference": returns,
                    }
                ).to_csv(output / "analysis" / f"{task}_h2_pairs.csv", index=False)
        merged.to_csv(output / "analysis" / f"{task}_returns.csv", index=False)
    table = pd.DataFrame(records)
    table["p_holm"] = np.nan
    within = table.stratum != "all"
    table.loc[within, "p_holm"] = holm(table.loc[within, "p"])
    table.to_csv(output / "analysis" / "h2.csv", index=False)
    pd.DataFrame(behavior).to_csv(output / "analysis" / "behavioral_summary.csv", index=False)
    write_json(output / "analysis" / "h2_exclusions.json", exclusions)
    return {"h2_tests": len(table), "h2_excluded_count": len(exclusions)}


def dispatch(args, frame, output, workflow):
    """Execute one authorized workflow and return serializable result metadata."""
    from src.config import DSA_WORKFLOWS, EVALUATION, TRAINING

    if workflow == "select_reservoirs":
        return select(args, output)
    if workflow == "plot_context_embeddings":
        from src.context_embeddings import run_embeddings

        return run_embeddings(args, frame, output)
    if workflow in TRAINING:
        from src.figures import plot_training
        from src.training import train_batch

        registry = train_batch(args, frame, output, workflow == "train_and_plot", workflow == "train_rl")
        if registry.empty:
            raise ValueError("No training configurations selected")
        plot_training(registry, output / "figures" / "learning_curves.png", args.dpi, workflow == "train_and_plot")
        return {"model_count": len(registry)}
    if workflow in EVALUATION:
        from src.evaluation import evaluate_registry
        from src.training import load_registry

        episodes = evaluate_registry(
            args, frame, load_registry(args, frame), output, workflow == "collect_obs_context_vec"
        )
        return {"episode_count": len(episodes)}
    if workflow == "random_scores":
        from src.evaluation import random_scores

        return {"episode_count": len(random_scores(args, output))}
    if workflow == "generate_fig_1":
        from src.figures import generate_motifs

        generate_motifs(args, output)
        return {"figure": "figures/wsbm_motifs.png"}
    if workflow in DSA_WORKFLOWS:
        result = analyze_distances(args, frame, output, workflow)
        if workflow == "analyze_hypotheses":
            result.update(analyze_h2(args, frame, output))
        return result
    if workflow in {"get_top_bottom_10_3_tasks", "descriptor_space_plot_performance"}:
        from src.analysis import performance_table, rank_groups

        scores = performance_table(args, frame)
        scores.to_csv(output / "analysis" / "performance.csv", index=False)
        for task in args.tasks:
            values = scores[scores.task == task].drop(columns="task")
            if workflow == "get_top_bottom_10_3_tasks":
                ranked = rank_groups(frame, values, args.group_size)
                ranked.to_csv(output / "analysis" / f"{task}_ranking.csv", index=False)
                print(ranked[["csv_idx", "stratum", "group", "performance"]].to_string(index=False))
            else:
                from src.figures import plot_performance_pca

                pool = pool_reference(args, required=True)
                merged = frame.merge(values, on="csv_idx", validate="one_to_one")
                if merged.empty:
                    raise ValueError(f"No {task} performance values found")
                plot_performance_pca(pool, merged, output / "figures" / f"{task}_performance_pca.png", args.dpi)
        return {"performance_count": len(scores)}
    raise ValueError(f"Unknown workflow: {workflow}")
