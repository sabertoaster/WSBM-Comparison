"""Solution 1 workflow: complete-cohort inference, figures and auditable summaries."""

import html

import numpy as np
import pandas as pd

from src.analysis import METRICS
from src.solution1_data import cached_distances, validate_sources
from src.solution1_stats import (
    PRIMARY,
    bootstrap_distances,
    correlations,
    exact_group_test,
    group_masks,
    interval,
    label_permutations,
    permutation_association,
    policy_pair_tensor,
    ranked_groups,
    regression_lines,
    seed_weights,
    structural_distances,
)
from src.utils import DESCRIPTORS, TASKS, write_json


def correct_families(table, interim):
    """Correct each prespecified family, counting undefined tests conservatively."""
    table = table.copy()
    table["p_holm_available"] = np.nan
    table["p_holm_final"] = np.nan
    for _, indices in table.groupby("test_family").groups.items():
        values = table.loc[indices, "p"].to_numpy(float)
        order = np.argsort(np.where(np.isfinite(values), values, 1))
        adjusted = np.maximum.accumulate(
            (len(values) - np.arange(len(values))) * np.where(np.isfinite(values), values, 1)[order]
        )
        result = np.empty(len(values))
        result[order] = np.minimum(adjusted, 1)
        result[~np.isfinite(values)] = np.nan
        table.loc[indices, "p_holm_available"] = result
        if not interim:
            table.loc[indices, "p_holm_final"] = result
        table.loc[indices, "family_tests_available"] = len(values)
    table["multiplicity_scope"] = "interim available tasks only" if interim else "all five tasks"
    return table


def ranking_summary(frame, per_seed, draws, size):
    """Summarize equal-weight seed returns and separately resampled group membership."""
    ids = frame.csv_idx.to_numpy()
    mean = per_seed.mean(axis=1)
    top, bottom = ranked_groups(mean, ids, size)
    memberships = np.zeros((len(draws), len(frame)), dtype=np.int8)
    for i, returns in enumerate(draws):
        a, b = ranked_groups(returns, ids, size)
        memberships[i, a], memberships[i, b] = 1, -1
    table = frame[["csv_idx", "stratum"]].copy()
    table["performance"] = mean
    table["policy_std"] = per_seed.std(axis=1, ddof=1) if per_seed.shape[1] > 1 else np.nan
    table["rank"] = 0
    table.loc[np.lexsort((ids, -mean)), "rank"] = np.arange(1, len(frame) + 1)
    table["group"] = "Middle"
    table.loc[top, "group"], table.loc[bottom, "group"] = "Top", "Bottom"
    table["top_frequency"] = (memberships == 1).mean(axis=0)
    table["bottom_frequency"] = (memberships == -1).mean(axis=0)
    for i in range(len(frame)):
        for key, value in interval(draws[:, i]).items():
            table.loc[i, key] = value
    return table, top, bottom


def pair_table(frame, returns, groups, structural, matrices):
    """Export all distinct pairs, including middle-performing reservoirs and every metric."""
    a, b = np.triu_indices(len(frame), 1)
    group = groups.group.to_numpy()
    labels = []
    for i, j in zip(a, b):
        labels.append(
            "TT"
            if group[i] == group[j] == "Top"
            else "BB"
            if group[i] == group[j] == "Bottom"
            else "TB"
            if {group[i], group[j]} == {"Top", "Bottom"}
            else "Other"
        )
    table = pd.DataFrame(
        dict(
            csv_idx_a=frame.csv_idx.to_numpy()[a],
            csv_idx_b=frame.csv_idx.to_numpy()[b],
            stratum_a=frame.stratum.to_numpy()[a],
            stratum_b=frame.stratum.to_numpy()[b],
            return_a=returns[a],
            return_b=returns[b],
            minimum_return=np.minimum(returns[a], returns[b]),
            group_a=group[a],
            group_b=group[b],
            pair_group=labels,
        )
    )
    for name, matrix in {**structural, **matrices}.items():
        table[name] = matrix[a, b]
    return table


def task_inference(args, frame, task, seed_returns, weights, structural, raw, aggregated, order, output):
    """Run all fixed and sensitivity tests with the same matched seed-block draws."""
    ids, seeds = frame.csv_idx.to_numpy(), args.policy_seeds
    bootstrap_returns = weights @ seed_returns.T
    size = 2 if args.smoke else 5
    rankings, top, bottom = ranking_summary(frame, seed_returns, bootstrap_returns, size)
    rankings.insert(0, "task", task)
    rankings.to_csv(output / "analysis" / f"{task}_ranking.csv", index=False)
    masks = group_masks(len(frame), top, bottom)
    permutations = {
        "unrestricted": label_permutations(len(frame), args.permutations, args.analysis_seed),
        "within_family": label_permutations(
            len(frame), args.permutations, args.analysis_seed, frame.stratum.to_numpy()
        ),
    }
    np.savez_compressed(output / "analysis" / f"{task}_permutation_labels.npz", **permutations)
    pairs = pair_table(frame, seed_returns.mean(axis=1), rankings, structural, aggregated)
    pairs.insert(0, "task", task)
    pairs.to_csv(output / "analysis" / f"{task}_pairs.csv", index=False)
    h1, groups, structure, band_tables = [], [], [], []
    upper = np.triu_indices(len(frame), 1)
    bootstrap_minimum = np.minimum(bootstrap_returns[:, upper[0]], bootstrap_returns[:, upper[1]])
    np.savez_compressed(
        output / "analysis" / f"{task}_bootstrap_returns.npz",
        returns=bootstrap_returns,
        csv_idx=ids,
        policy_seeds=np.asarray(seeds),
    )
    for metric in METRICS:
        print(f"Inference {task}: {metric}", flush=True)
        primary = "primary" if metric in PRIMARY else "exploratory"
        tensor = policy_pair_tensor(raw[metric], order, ids, seeds)
        distance_draws = bootstrap_distances(tensor, weights)
        if not np.allclose(tensor.mean(axis=(1, 2)), aggregated[metric][upper]):
            raise ValueError("Pair tensor and all-cross-policy aggregation disagree")
        h1_draws = correlations(bootstrap_minimum, distance_draws)
        for scheme, labels in permutations.items():
            h1.append(
                dict(
                    task=task,
                    metric=metric,
                    role=primary,
                    permutation=scheme,
                    test_family=f"h1_continuous_{primary}_{scheme}",
                    n_reservoirs=len(frame),
                    n_pairs=len(pairs),
                    **permutation_association(seed_returns.mean(axis=1), aggregated[metric], labels),
                    **interval(h1_draws),
                )
            )
        summary = exact_group_test(aggregated[metric], top, bottom)
        draws = {key: distance_draws[:, mask].mean(axis=1) for key, mask in masks.items()}
        draws["delta"] = draws["BB"] - draws["TT"]
        row = dict(
            task=task,
            metric=metric,
            role=primary,
            test_family=f"h1_groups_{primary}",
            top_ids=",".join(map(str, ids[top])),
            bottom_ids=",".join(map(str, ids[bottom])),
            TT_pairs=int(masks["TT"].sum()),
            BB_pairs=int(masks["BB"].sum()),
            TB_pairs=int(masks["TB"].sum()),
            **summary,
        )
        for key, values in draws.items():
            row.update({key + "_" + k: v for k, v in interval(values).items()})
        groups.append(row)
        saved = dict(distances=distance_draws, h1_rho=h1_draws, **draws)
        for structural_metric, matrix in structural.items():
            predictor = matrix[upper]
            rho_draws = correlations(predictor, distance_draws)
            saved[structural_metric + "_rho"] = rho_draws
            for scheme, labels in permutations.items():
                structure.append(
                    dict(
                        task=task,
                        metric=metric,
                        role=primary,
                        structural_metric=structural_metric,
                        permutation=scheme,
                        test_family=f"structure_{primary}_{structural_metric}_{scheme}",
                        n_reservoirs=len(frame),
                        n_pairs=len(pairs),
                        **permutation_association(None, aggregated[metric], labels, matrix),
                        **interval(rho_draws),
                    )
                )
            grid = np.linspace(predictor.min(), predictor.max(), 100)
            lines = regression_lines(predictor, distance_draws, grid)
            observed_line = regression_lines(predictor, aggregated[metric][upper][None, :], grid)[0]
            band = pd.DataFrame(
                dict(
                    task=task,
                    metric=metric,
                    structural_metric=structural_metric,
                    structural_distance=grid,
                    descriptive_ols=observed_line,
                )
            )
            for index in range(len(grid)):
                for key, value in interval(lines[:, index]).items():
                    band.loc[index, key] = value
            band_tables.append(band)
            saved[structural_metric + "_line_grid"] = grid
            saved[structural_metric + "_lines"] = lines
        np.savez_compressed(output / "analysis" / f"{task}_{metric}_bootstrap.npz", **saved)
    bands = pd.concat(band_tables, ignore_index=True)
    bands.to_csv(output / "analysis" / f"{task}_regression_bands.csv", index=False)
    return h1, groups, structure, rankings, pairs, bands


def render_report(output, tables, provenance):
    """Write a standalone HTML index with explicit scope, deviations and linked figure sources."""
    intro = (
        "Closed-loop associations in the selected cohort. Different reservoirs may have different read-in matrices, "
        "and policies generate their own inputs. This analysis does not establish causal connectivity effects, "
        "an optimal manifold, held-out identification accuracy, or structural irrelevance from a null result."
    )
    content = [
        "<!doctype html><html><meta charset='utf-8'><title>Solution 1 analysis</title>",
        "<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}table{border-collapse:collapse;font-size:12px}"
        "td,th{padding:5px;border:1px solid #ccc}img{max-width:100%}</style>",
        f"<h1>Solution 1: {html.escape(provenance['coverage'])}</h1><p>{intro}</p>",
        "<p>" + html.escape(provenance["interval_scope"]) + ". Five training-seed blocks provide limited precision. "
        "Intervals exclude reservoir-population, episode-sampling and identification uncertainty.</p>",
        "<p>Holm families are separate for continuous H1, exact secondary H1, Euclidean Fig. 3B, "
        "within-family sensitivity and Mahalanobis sensitivity. Other dynamics metrics remain exploratory. "
        "Interim corrections cover available tasks only; final five-task adjusted values remain blank.</p>",
    ]
    for deviation in provenance["deviations"]:
        content.append("<p><strong>Protocol deviation: " + html.escape(deviation) + "</strong></p>")
    content.append("<p>Actual training regimes (n_envs × n_steps changes rollout samples and PPO update timing):</p>")
    content.append(pd.DataFrame(provenance["training_regimes"]).to_html(index=False))
    for name, table in tables.items():
        primary = table[table.role == "primary"]
        content.extend(
            [
                f"<h2>{html.escape(name)}</h2><p><a href='analysis/{name}.csv'>All results and source table</a></p>",
                primary.to_html(index=False, float_format=lambda x: f"{x:.5g}"),
            ]
        )
    content.append(
        "<h2>Figures</h2><p>PNG and PDF versions have adjacent CSV source tables in analysis/. "
        "Learning curves use linear interpolation only inside the common observed interval; confidence bands are pointwise. "
        "Family summaries average the four reservoir means equally and do not represent the wider sampling population. "
        "Group effect intervals freeze observed membership; ranking stability reranks each bootstrap draw.</p>"
    )
    for path in sorted((output / "figures").glob("*.png")):
        content.append(
            f"<p><a href='figures/{path.stem}.pdf'>{html.escape(path.stem)} (PDF)</a></p><img src='figures/{path.name}'>"
        )
    content.append(
        "<p>The distinct manuscript H2, structure versus absolute return difference, is saved separately in analysis/h2.csv. "
        "It is not the Fig. 3B structure–dynamics test.</p></html>"
    )
    (output / "report.html").write_text("\n".join(content))


def run_solution1(args, frame, output):
    """Execute the corrected analysis without retraining or silently recollecting data."""
    from src.figures import plot_solution1_inference, plot_solution1_performance
    from src.workflows import analyze_h2, pool_reference

    frame = frame.reset_index(drop=True)
    ranking, dynamics, curves, provenance = validate_sources(args, frame, output)
    pool = pool_reference(args, required=True)
    structural, parameters = structural_distances(
        frame[list(DESCRIPTORS)].to_numpy(float), pool[list(DESCRIPTORS)].to_numpy(float)
    )
    write_json(output / "analysis" / "descriptor_standardization.json", parameters)
    np.savez_compressed(
        output / "analysis" / "structural_distances.npz", **structural, csv_idx=frame.csv_idx.to_numpy()
    )
    weights, draws = seed_weights(args.bootstrap_replicates, len(args.policy_seeds), args.analysis_seed)
    np.savez_compressed(
        output / "analysis" / "bootstrap_seed_blocks.npz",
        weights=weights,
        label_indices=draws,
        policy_seeds=np.asarray(args.policy_seeds),
        sampled_labels=np.asarray(args.policy_seeds)[draws],
    )
    per_seed = ranking.groupby(["task", "csv_idx", "policy_seed"], as_index=False)["return"].mean()
    per_seed.to_csv(output / "analysis" / "policy_seed_returns.csv", index=False)
    h1, groups, structure = [], [], []
    for task in args.tasks:
        print(f"Solution 1 fitting {task}: {len(frame) * len(args.policy_seeds)} policy systems", flush=True)
        raw, aggregated, order, _ = cached_distances(args, frame, task, output, dynamics)
        seed_returns = (
            per_seed[per_seed.task == task]
            .pivot(index="csv_idx", columns="policy_seed", values="return")
            .loc[frame.csv_idx, args.policy_seeds]
            .to_numpy()
        )
        task_h1, task_groups, task_structure, rankings, pairs, bands = task_inference(
            args, frame, task, seed_returns, weights, structural, raw, aggregated, order, output
        )
        h1.extend(task_h1)
        groups.extend(task_groups)
        structure.extend(task_structure)
        plot_solution1_performance(task, frame, seed_returns, weights, curves, args.policy_seeds, output, args.dpi)
        plot_solution1_inference(task, rankings, pairs, bands, pd.DataFrame(task_groups), output, args.dpi)
    interim = set(args.tasks) != set(TASKS) or args.smoke
    tables = {
        "h1_continuous": correct_families(pd.DataFrame(h1), interim),
        "h1_groups": correct_families(pd.DataFrame(groups), interim),
        "structure_dynamics": correct_families(pd.DataFrame(structure), interim),
    }
    for name, table in tables.items():
        table["result_scope"] = "exploratory/deviations" if provenance["deviations"] else provenance["coverage"]
        table["interval_scope"] = provenance["interval_scope"]
        table.to_csv(output / "analysis" / f"{name}.csv", index=False)
    h2 = analyze_h2(args, frame, output)
    render_report(output, tables, provenance)
    write_json(output / "analysis" / "protocol_status.json", provenance)
    return dict(
        solution1_schema=3,
        task_count=len(args.tasks),
        reservoir_count=len(frame),
        policy_systems_per_task=len(frame) * len(args.policy_seeds),
        report="report.html",
        confirmatory_eligible=provenance["confirmatory_eligible"],
        deviations=provenance["deviations"],
        **h2,
    )
