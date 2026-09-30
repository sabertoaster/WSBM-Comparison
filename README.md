# WSBM reservoirs and InputDSA experiments

This repository contains the experiments for *Analyzing the Dynamics of WSBM-based Echo State Networks using InputDSA*. It samples reservoir structures, trains developmental PPO policies, collects reservoir trajectories, and compares recurrent and input-driven dynamics. No evolutionary optimization is run.

Start with the examples below. [The workflow reference](docs/WORKFLOWS.md) documents every script and its parameters. [The discrepancy register](docs/DISCREPANCIES.md) explains where the historical implementation differs from the supplied manuscript and which results require new experiments.

## Installation

Use Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). From this repository:

```sh
uv sync --locked
uv run --locked python main.py --help
```

`pyproject.toml` retains the compatible ReservoirPy fork, and `uv.lock` pins its existing commit. DSA is installed as an editable local dependency from `DSA`, whose Python package is `DSA/DSA`. Keep that external checkout present. This repository tracks DSA as a Git link without `.gitmodules`; a fresh clone does not automatically fetch it. Obtain the matching external checkout at the commit listed in [the discrepancy register](docs/DISCREPANCIES.md) before syncing. Do not replace an already populated DSA directory.

The project installs `src`, the script entry points, and the `er_mrl` compatibility namespace. New Python code should import `src.er_mrl`. Tests and Ruff are included in the default development group. `requirements.txt` is retained as a pip installation alternative; uv plus the lockfile is the reproducible installation path.

MuJoCo runs without a display for these training/evaluation workflows. Keep the paper's `v4` environment IDs for historical comparisons; upgrading environment versions would change the experiment. CPU is the default. TensorBoard uses its bundled TensorFlow stub for PPO scalar logging, avoiding an unnecessary TensorFlow CUDA runtime.

## Choose the protocol

**Legacy** reads existing checkpoints, root `selected.csv`, `trajectories`, and `rl_only`. It preserves historical reservoir/input conventions while fixing CLI, path, and reporting issues.

**Corrected** uses the same saved reservoir matrix for selection, training, and evaluation; masks velocities; encodes actual actions; resets context per episode; and separates policy seeds from structure seeds. New corrected selections are required. Both historical CSVs and all existing figures stay in place.

The default corrected design is **100 reservoir configurations, with 200 neurons in each reservoir**. These are separate quantities: five strata × 20 selected configurations gives 100 rows. Batch training defaults to five policy seeds per reservoir and 500,000 steps per seed.

Every experiment requires `--protocol`. A dry run validates and prints settings without training or writing:

```sh
uv run --locked python scripts/inputdsa_100_reservoirs_H1.py \
  --protocol legacy --tasks Swimmer-v4 --dry-run
```

## Corrected experiment sequence

The following commands select structures, train policies and a matched PPO baseline, collect ten deterministic episodes per policy, and test both paper hypotheses. These are full experiment commands; use the small check below first.

```sh
uv run --locked python scripts/select_reservoirs.py \
  --protocol corrected --run-id selection

uv run --locked python scripts/train_all_selected.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection/selection/selected.csv \
  --run-id training

uv run --locked python scripts/collect_obs_context_vec.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection/selection/selected.csv \
  --models-csv artifacts/corrected/training/models/models.csv \
  --run-id collection

uv run --locked python scripts/analyze_hypotheses.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection/selection/selected.csv \
  --evaluation-csv artifacts/corrected/collection/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/collection/trajectories/trajectories.csv \
  --run-id hypotheses
```

Selection uses only eight structural descriptors and stratified MaxMin diversity. Training cannot affect the selection. Paper H1 compares within-top and within-bottom dynamical distances; paper H2 compares descriptor distance with absolute mean-return difference. Tests permute reservoir labels to preserve dependence among pairwise distances. Their new p-values need not match the manuscript's historical values.

## Small end-to-end check

This sequence uses 20-neuron reservoirs, four selected configurations, one policy seed, 64 training steps, one episode, and 19 permutations. Results verify execution and data alignment, not the paper's hypotheses. Use unused run names for each retry; no cleanup is required.

```sh
uv run --locked python scripts/select_reservoirs.py \
  --protocol corrected --smoke --run-id smoke-selection

uv run --locked python scripts/train_all_selected.py \
  --protocol corrected --smoke --tasks Swimmer-v4 \
  --selected-csv artifacts/corrected/smoke-selection/selection/selected.csv \
  --run-id smoke-training

uv run --locked python scripts/collect_obs_context_vec.py \
  --protocol corrected --smoke --tasks Swimmer-v4 \
  --selected-csv artifacts/corrected/smoke-selection/selection/selected.csv \
  --models-csv artifacts/corrected/smoke-training/models/models.csv \
  --run-id smoke-collection

uv run --locked python scripts/analyze_hypotheses.py \
  --protocol corrected --smoke --tasks Swimmer-v4 \
  --selected-csv artifacts/corrected/smoke-selection/selection/selected.csv \
  --evaluation-csv artifacts/corrected/smoke-collection/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/smoke-collection/trajectories/trajectories.csv \
  --run-id smoke-analysis
```

`--smoke` performs work; `--dry-run` does not. The shell launcher `./run_test.sh` provides a small direct legacy training check.

## Existing results and figures

```sh
uv run --locked python scripts/dev_scripts.py \
  --protocol legacy Swimmer-v4 --backend dmdc --run-id legacy-swimmer

uv run --locked python scripts/generate_fig_1.py \
  --protocol legacy --run-id motif-figure
```

The root selection and `scripts/selection/selected.csv` differ. Select the latter explicitly if that dataset produced your checkpoints. No command searches alternate CSVs automatically. Legacy `--backend dmdc` retains the historical custom-subspace routing; corrected `dmdc` uses the external standard DMDc class.

Original script names remain supported. Their historical H1/H2 suffixes are explained in [the script table](docs/WORKFLOWS.md#rankings-and-dynamics). Existing `myplot.png`, selected data, old analysis figures, models, and trajectories are not overwritten by default.

## Outputs and project layout

New runs write `artifacts/PROTOCOL/RUN_ID/manifest.json` and the relevant subdirectories:

| Directory | Contents |
| --- | --- |
| `selection` | Candidate descriptors, selected rows, rejection reasons, PCA, corrected matrices |
| `models` | PPO ZIPs, environment configurations, model registry |
| `logs` | TensorBoard events, separated by configuration and policy seed |
| `evaluation` | Episode returns, lengths, termination flags, replicate summaries |
| `trajectories` | Episode NPZ files and registry |
| `analysis` | Ordered distance matrices, fitting diagnostics, pair tables, H1/H2 tests, exclusions |
| `figures` | Saved headless figures |

Manifests record resolved options, dependency versions, code and DSA revisions, source-file and input-CSV fingerprints, and completion/failure status. Model registries record recurrent and input-weight hashes; evaluation verifies both. Explicit output paths are relative to the caller; defaults are relative to this repository. Reusing a run ID fails unless `--overwrite` is supplied. Nothing is deleted automatically.

`scripts` contains thin executable entry points. `src/utils.py` supplies common artifact, CSV, rank, and TensorBoard helpers. Other `src` modules implement structural factories, selection, training, evaluation, analysis, and plotting. `src/er_mrl` preserves adapted wrappers and developmental helpers. `src/wsbm_esn.py` retains the standalone reference ESN implementation. `DSA` remains external. [Validation instructions](docs/VALIDATION.md) cover the tests and smoke pipeline.
