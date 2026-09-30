# Research workflow reference

This reference covers every first-party executable and explains how data moves from structural selection to the paper's analyses. Start with the installation and complete examples in [README](../README.md). The mathematical and historical differences are recorded in [the discrepancy register](DISCREPANCIES.md).

## Shared command interface

Every experiment requires `--protocol legacy` or `--protocol corrected`. All scripts expose `build_parser()` and `main(argv=None)` and can be imported without starting an experiment. `python main.py WORKFLOW ...`, `wsbm WORKFLOW ...`, and `python scripts/WORKFLOW.py ...` call the same implementation. Use `--help` on an individual workflow for its parameters.

| Option | Meaning |
| --- | --- |
| `--selected-csv FILE` | Explicit selection; legacy defaults to repository root `selected.csv`. Corrected runs require a corrected selection. |
| `--tasks ID ...` | Tasks to process. Corrected defaults are Hopper-v4, Walker2d-v4, Swimmer-v4, Ant-v4, and HalfCheetah-v4. |
| `--limit N` | First N original configurations; stable `csv_idx` survives sorting and filtering. |
| `--output-dir DIR` | Artifact root. Default is this repository's `artifacts`; explicit paths are relative to the caller. |
| `--run-id NAME` | Directory name under `artifacts/PROTOCOL`. Without it, a UTC timestamp and random suffix create a new run. |
| `--overwrite` or `--force` | Permit writing into an existing run directory. Nothing is deleted; use a new ID for an independent experiment. |
| `--dry-run` | Validate arguments and selection, print resolved settings, and exit without writing or executing. |
| `--smoke` | Execute a small check: one task, up to four configurations, one policy seed, 64 training steps, one evaluation episode, and 19 permutations. |
| `--workers N` | Parallel workers for structural sampling and DSA fitting. Default 1 avoids oversubscribing a workstation. |
| `--dpi N` | Figure export resolution. Figures use a headless backend. |

Smoke selection uses five strata, five candidates per stratum, one pick per stratum, 20 neurons, and community counts 2 or 3. Other smoke stages infer the saved selection's neuron count. Smoke settings deliberately override experiment budgets; omit `--smoke` for scientific runs.

Historical underscore spellings remain accepted for existing training flags, such as `--env_id`, `--training_steps`, `--res_lr`, and `--use_reservoir True`. Boolean options also support `--flag`, `--flag False`, and `--no-flag`.

## Structural selection

`select_reservoirs.py` implements sample → describe → standardize → PCA → stratified MaxMin selection. It never reads training rewards or DSA distances.

```sh
python scripts/select_reservoirs.py --protocol corrected --run-id selection \
  --pool-per-stratum 1000 --select-per-stratum 20 --units 200 --selection-seed 0
```

The 100 configurations are five groups of 20. `--units 200` means 200 neurons in each configuration. The full default pool contains 5,000 candidates before feasibility checks. A stratum with fewer valid candidates than requested fails with a recorded rejection table; increase the pool instead of silently reducing its selected count.

Sampling controls include `--strata`, `--communities`, `--density-range`, `--centre-range`, `--dispersion-range`, `--topology-range`, `--contrast-range`, `--size-alphas`, `--core-range`, and `--mixing-range`. Each range accepts two endpoints. The null arm sets contrast to zero, topology and dispersion ratios to one, and equal community sizes. `--null-centre-range` defaults to 2–10; `--core-contrast-range` and `--mixed-contrast-range` default to 1–8.

Selection exports `selection/descriptors.csv`, `selection/selected.csv`, `selection/rejections.csv`, `selection/pca.npz`, and `figures/selection.png`. Corrected selections also export `selection/matrices/reservoir_ID.npz`. Relative `matrix_file` values resolve against the selected CSV's directory. Keep that directory together when transferring an experiment.

## Training and baseline comparisons

| Script | Function | Legacy defaults |
| --- | --- | --- |
| `train_rl.py` | One direct reservoir/PPO configuration; corrected mode selects one `--csv-idx` | Required task; 300,000 steps; 100 neurons; reservoir enabled; no baseline |
| `train_all_selected.py` | All selected configurations plus matched PPO | Ant, HalfCheetah, Swimmer; 500,000 steps; one run per CSV seed |
| `train_and_plot.py` | First actual row per stratum and replicate reward curves | Same three tasks; five seeds; both reservoir and policy seeds vary |

Corrected training defaults to five policy seeds `0 1 2 3 4`, 500,000 steps for batch training, and 200 neurons from a full-size corrected selection. The saved selection determines reservoir seed and neuron count when `--units` is omitted. Input weights use the fork's dense Gaussian convention with the structure seed and configured input scaling. Policy seeds do not alter the recurrent matrix or input weights. PPO baselines receive the same observation mask as reservoir agents.

```sh
python scripts/train_rl.py --protocol corrected --env-id Swimmer-v4 \
  --selected-csv artifacts/corrected/selection/selection/selected.csv --csv-idx 0 \
  --policy-seeds 0 1 2 3 4 --training-steps 500000 --run-id swimmer-row0
```

Reservoir controls are `--res-lr`, `--res-sr`, `--res-iss`, `--skip-c`, and `--reset-res`. Corrected runs reject a spectral radius or neuron count inconsistent with the saved matrix. `--del-obs` enables the velocity mask; corrected default is enabled. Turning it off explicitly creates a fully observable experiment and records that choice.

PPO options include `--learning-rate`, `--n-envs`, `--n-steps`, `--batch-size`, `--n-epochs`, `--gamma`, `--gae-lambda`, `--clip-range`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`, and `--device`. PPO collects complete rollout batches, so actual training timesteps can exceed the requested budget. `--environment-seed` controls the simulator independently of policy initialization. `--max-episode-steps` changes the time limit and is saved in the model configuration.

Direct legacy training additionally accepts motif parameters `--motif`, `--n-communities`, `--hi`, `--lo`, `--mid`, `--sigma`, `--connectivity`, `--symmetric`, and `--p-negative`. These describe a direct model; selected-batch training uses the CSV conversion instead.

Training exports model ZIPs, per-model JSON environment configurations, `models/models.csv`, TensorBoard events under `logs`, and `figures/learning_curves.png`. Curves use all events and align replicates by recorded timestep. Mean ± standard deviation is descriptive, not a confidence interval.

## Evaluation and trajectory collection

`evaluate.py` evaluates deterministic actions using shared episode reset seeds. `collect_obs_context_vec.py` performs the same evaluation and records reservoir inputs and states. Both read a training registry through `--models-csv` and export `evaluation/episodes.csv` plus per-policy `evaluation/returns.csv`.

```sh
python scripts/collect_obs_context_vec.py --protocol corrected \
  --selected-csv artifacts/corrected/selection/selection/selected.csv \
  --models-csv artifacts/corrected/training/models/models.csv \
  --evaluation-episodes 10 --evaluation-seed 10000 --run-id collection
```

Each corrected NPZ contains `states` and `inputs` shaped `(time, features)`, identifiers, and protocol metadata. A scalar Gymnasium rollout avoids automatic resets. For DMDc, `inputs[t]` is the forcing that advances `states[t]` to `states[t+1]`; the final unused input repeats the preceding input to retain equal array lengths. Each episode is a separate trial. Replicates are additional trials of the same fixed reservoir, not additional reservoirs in the hypothesis test.

The trajectory registry is `trajectories/trajectories.csv`. Use it for corrected DSA; paths identify individual episodes. Historical `Ot.npy` and `Ct.npy` arrays remain readable through `--trajectories-dir`. Legacy arrays lack reliable episode-boundary metadata and retain the original input/post-state alignment.

For old checkpoints, omit `--models-csv`, select legacy mode, and use `--models-dir`, `--logs-dir`, `--training-steps`, and the exact selection used to train them. This reader recognizes the all-selected naming pattern. Other checkpoint layouts should be represented with an explicit registry containing saved configuration paths.

`src/er_mrl/get_random_scores.py` evaluates random-action policies:

```sh
python -m src.er_mrl.get_random_scores --protocol legacy --tasks Swimmer-v4 \
  --evaluation-episodes 100 --run-id random-baseline
```

It exports total episode return and mean reward per step separately; the historical random-score constants used the latter quantity. Legacy defaults to the original eleven MuJoCo tasks; corrected defaults to the five paper tasks.

## Rankings and dynamics

`get_top_bottom_10_3_tasks.py` writes performance and top/bottom tables. `descriptor_space_plot_performance.py` projects selected descriptors through full-pool PCA and colors them by performance. Choose `--ranking-source evaluation --evaluation-csv FILE` or `--ranking-source tensorboard`. Corrected defaults to evaluation; legacy defaults to TensorBoard with a linear last-10,000-step score. `--weight-type` accepts linear, exponential, or mean. Rankings exclude missing/nonfinite rewards and reject overlapping top/bottom groups.

| Script | Analysis | Legacy task/backend defaults |
| --- | --- | --- |
| `inputdsa_100_reservoirs_H1.py` | Full-subset structural/state scatter and hexbin | Humanoid-v4; historical dmdc alias |
| `inputdsa_top_bottom_10_reservoirs_H1.py` | Structural/state plots for performance extremes | Walker2d-v4; historical dmdc alias |
| `inputdsa_100_reservoirs_H2.py` | Five distance matrices across all selected reservoirs | Ant, HalfCheetah, Swimmer; N4SID |
| `dev_scripts.py` | Five matrices for top/bottom groups and paper-H1 summaries | Five paper tasks; N4SID |
| `inputdsa_5_systems_mackey_glass_ou.py` | One reservoir per stratum under synthetic input | 2,000 samples; 10 delays; N4SID; minimum rank 1 |
| `inputdsa_100_reservoirs_mackey_glass_ou.py` | All selected reservoirs under synthetic input | 3,000 samples; 10 delays; N4SID; minimum rank 3 |

All corrected dynamics workflows default to the explicit DSA `DMDc` class. N4SID requires `--backend n4sid`. No numerical failure changes backend automatically. The legacy `dmdc` string reproduces the historical custom-subspace routing, as explained in the discrepancy register.

`--n-delays`, `--rank`, `--rank-energy`, `--min-rank`, `--max-rank`, and `--dmd-regularization` control fitting. A common rank is chosen from delay-embedded state variance and clipped to supported dimensions; diagnostics record per-system ranks and caps. Constant, short, or nonfinite trajectories fail rather than producing arbitrary distances. `--state-metric`, `--state-iters`, and `--state-learning-rate` control the separate state comparison. Its historical default is Wasserstein distance; joint comparisons and separate control comparison use the external controllability implementation.

Synthetic workflows expose `--conditions mackey_glass ou_noise`, `--input-steps`, `--input-dim`, `--input-seed`, and `--ou-theta`. OU dimension affects OU input only. The same seeded input drives every reservoir; corrected matrices are reused across input dimensions. Legacy synthetic leak rate remains 0.3, unlike legacy RL's 0.1.

Distance workflows export NPZ matrices, ordered row CSVs, fitting diagnostics, and five-panel figures. Diagnostics include state/control operator norms, conditioning, and control singular values. Missing trajectories fail by default. `--allow-subset` permits explicit exclusions and writes their IDs; filtered descriptors and trajectories remain aligned. H1 still requires at least two complete reservoirs in each group.

## Paper hypothesis tests

`analyze_hypotheses.py` runs both paper hypotheses:

```sh
python scripts/analyze_hypotheses.py --protocol corrected \
  --selected-csv artifacts/corrected/selection/selection/selected.csv \
  --evaluation-csv artifacts/corrected/collection/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/collection/trajectories/trajectories.csv \
  --group-size 10 --permutations 9999 --analysis-seed 0 --run-id hypotheses
```

H1 ranks reservoirs by mean deterministic return, averaging episodes within each policy seed and then averaging policy seeds equally. It reports top and bottom within-group means for five metrics. The statistic is bottom mean minus top mean; a one-sided group-label permutation test measures evidence for lower top-group distances. Holm adjustment covers the requested task/metric family.

H2 calculates Spearman correlation between standardized eight-dimensional descriptor distance and absolute mean-return difference. Corrected standardization uses `--pool-csv` or the selection's sibling `descriptors.csv`; the pool must have matching protocol, neuron count, and selected matrix hashes. Permutations shuffle whole reservoir return labels, preserving dependence between pairs. Global and within-stratum results appear in `analysis/h2.csv`; Holm adjustment covers all requested within-stratum task tests. Fewer than three reservoirs or constant distances produce an explicit undefined result. H2 always requires deterministic episode returns, even when running legacy data.

Per-task pair tables and return tables accompany plots. `analysis/behavioral_summary.csv` reports the return range, mean, standard deviation, coefficient of variation, and policy-replicate counts per task. The code does not infer that a nonsignificant correlation proves structure is irrelevant, and smoke results are not scientific evidence.

## Figures and preserved helpers

`generate_fig_1.py` draws canonical assortative, disassortative, core–periphery, and mixed graphs. Legacy uses the local reference ESN implementation with 68 neurons, six communities, and seed 42. Corrected uses the canonical structural factory. The output is `figures/wsbm_motifs.png` inside a new run; existing `myplot.png` remains unchanged.

Figure controls include `--units`, `--n-communities`, `--figure-seed`, `--hi`, `--lo`, `--mid`, `--connectivity`, `--core-fraction`, and `--sigma-base`. The default `--layout community` uses the preserved Netgraph community layout; `--layout spring` provides a NetworkX alternative. All panels share one absolute-weight color scale.

`src/er_mrl/wrappers.py` retains the multi-reservoir, reward-saving, partially observable, and collection wrappers for existing callers. The new research pipeline uses `ReservoirWrapper` and `DeletedVelocityWrapper`. `src/er_mrl/utils.py` retains interactive observation/context/FFT plots and asks for explicit interactive confirmation in its manual deletion helper; no workflow invokes deletion. `src/er_mrl/experiments.py` retains historical Optuna helpers and benchmark constants, and the research CLI never runs evolutionary optimization.

`run_test.sh` launches a small legacy Swimmer training check using this checkout's virtual environment. Additional arguments are forwarded. `main.py` lists all workflows, and `er_mrl` forwards old imports to `src.er_mrl`.
