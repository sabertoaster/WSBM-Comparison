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

The following commands implement [Solution 1](rebuttal/solution1.md): 20 structures,
five policy seeds, separate ranking and dynamics episodes, H1 and Fig. 3B inference,
and the distinct manuscript H2. These are full experiments; use the small check
below first. Other selection workflows retain the default 100 configurations.

```sh
uv run --locked python scripts/select_reservoirs.py \
  --protocol corrected --run-id selection-20 \
  --pool-per-stratum 1000 --select-per-stratum 4 --pca-components 8 --units 200

uv run --locked python scripts/train_all_selected.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --run-id training-20-seeds5

uv run --locked python scripts/collect_obs_context_vec.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --models-csv artifacts/corrected/training-20-seeds5/models/models.csv \
  --evaluation-seed 10000 --evaluation-episodes 10 --run-id ranking-20-seeds5

uv run --locked python scripts/collect_obs_context_vec.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --models-csv artifacts/corrected/training-20-seeds5/models/models.csv \
  --evaluation-seed 20000 --evaluation-episodes 10 --run-id dynamics-20-seeds5

uv run --locked python scripts/analyze_hypotheses.py \
  --protocol corrected --backend dmdc \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --evaluation-csv artifacts/corrected/ranking-20-seeds5/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/dynamics-20-seeds5/trajectories/trajectories.csv \
  --run-id hypotheses-solution1-20-seeds5
```

Selection uses only eight structural descriptors and stratified MaxMin diversity.
Primary H1 correlates each pair's minimum reservoir return with dynamical distance,
using all 190 pairs. Secondary H1 compares top-five and bottom-five distances with
all 252 conditional group allocations. Fig. 3B tests structure versus dynamics;
manuscript H2 separately tests structure versus absolute return difference.

Corrected trajectory analyses default to SubspaceDMDc/N4SID; use `--backend dmdc`
to reproduce the original Solution 1 identification. Both fit one system per reservoir and policy seed.
Each episode's source/target transitions are built separately before concatenation,
so episode resets never become fitted transitions. With 20 reservoirs and five
policy seeds, this fits 100 systems per task and chooses one common rank across
all 100 before selecting top/bottom groups. Reservoir-pair distances average all
25 cross-policy combinations with equal weight.

Analysis exports `<task>_policy_distances.npz` and `<task>_policy_order.csv` for
the individual fitted systems, `<task>_reservoir_distances.npz` for the complete
aggregated cohort. The other dynamics scripts also export `<task>_distances.npz`
for their ordered analysis groups. Diagnostics record system identities, episode transition counts, rank,
and aggregation settings. Missing policies, duplicate/incomplete episodes, or
inconsistent read-in hashes halt analysis. Defaults require policy seeds 0–4 and
ten episodes each. Solution 1 requires the complete fixed cohort; other dynamics
scripts support explicitly different `--policy-seeds` and `--dynamics-episodes`.
`--smoke` uses one seed and one episode and produces validation results only.

The analysis saves `report.html`, all five metric matrices, pair and ranking
tables, diagnostics, permutation labels, and bootstrap distributions. Figures
include all-reservoir learning curves, all seed evaluation means, supplementary
family summaries, continuous H1, fixed-group raw distances, and Euclidean and
Mahalanobis structure–dynamics plots. Every figure has CSV sources; publication
figures have PNG and PDF versions. The 5,000 matched seed-block bootstrap draws
retain repeated labels on both distance axes and use the same draws across tasks.
Intervals are conditional on the selected structures and collected episodes.
Primary metrics are `state_joint` and `control_joint`; other metrics are exploratory.
Unrestricted, within-family, secondary-group, and Mahalanobis tests have separate
Holm families. Interim adjustments are labelled and final five-task values remain
blank until all tasks are present.

Reuse trained policies and the existing ranking evaluation at seeds 10000–10009.
Collect new dynamics episodes at 20000–20009, then refit old pooled or boundary-
contaminated results. No RL retraining is required for these analysis additions.
Checkpoint and encoder hashes must match both collections. Imported VM paths
resolve under the local `artifacts` root without changing registry files.

For the downloaded Hopper/Walker cohort, run these commands on the VM or locally:

```sh
uv run --locked python scripts/collect_obs_context_vec.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --models-csv artifacts/corrected/training-20-seeds5/models/models.csv \
  --tasks Hopper-v4 Walker2d-v4 --policy-seeds 0 1 2 3 4 \
  --evaluation-seed 20000 --evaluation-episodes 10 --device cpu \
  --run-id dynamics-20-seeds5-hopper-walker

uv run --locked python scripts/analyze_hypotheses.py \
  --protocol corrected --backend dmdc --interim \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --evaluation-csv artifacts/corrected/collection-20-seeds5-hopper-walker/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/dynamics-20-seeds5-hopper-walker/trajectories/trajectories.csv \
  --tasks Hopper-v4 Walker2d-v4 --device cpu \
  --run-id hypotheses-solution1-20-seeds5-hopper-walker
```

`--interim` permits complete task subsets, never incomplete reservoir/policy
cohorts. Actual checkpoint settings are audited: the existing Walker cohort has
44 policies with 10 × 256 rollout samples and 56 with 20 × 256. These policies
are reused with the deviation disclosed; the report does not label them uniform
confirmatory training. Saved training settings and regime counts accompany results.

To explore existing ranking trajectories immediately, explicitly add
`--exploratory-reuse-ranking-episodes` and point `--trajectories-csv` at the old
collection. Outputs are labelled exploratory. This option does not satisfy the
separate dynamics-episode requirement. Use an unused run ID for each analysis.
`--distance-cache artifacts/corrected/<previous-solution1-run>` reuses fitted
matrices only when payload hashes, system identities, fitting settings, software
versions and fit source code match. Other statistics and figures are regenerated.

### SubspaceDMDc follow-up

The corrected default is now `--backend n4sid`, using the local DSA package's
QR-based SubspaceDMDc implementation. No `nfoursid` dependency is required.
Legacy backend defaults and the historical `dmdc` alias retain their routing.
N4SID identifies a latent realization; operator distances depend on its
coordinates and scaling. Changing the software default does not establish a
new confirmatory protocol: N4SID hypothesis results are labelled exploratory
and `confirmatory_eligible` is false.

On the machine with the complete original five-task collections, run:

```sh
uv run --locked python scripts/analyze_hypotheses.py \
  --protocol corrected --backend n4sid \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --pool-csv artifacts/corrected/selection-20/selection/descriptors.csv \
  --models-csv artifacts/corrected/training-20-seeds5/models/models.csv \
  --evaluation-csv artifacts/corrected/ranking-20-seeds5-alltasks/evaluation/episodes.csv \
  --trajectories-csv artifacts/corrected/collection-20-seeds5-alltasks/trajectories/trajectories.csv \
  --tasks Hopper-v4 Walker2d-v4 Swimmer-v4 Ant-v4 HalfCheetah-v4 \
  --policy-seeds 0 1 2 3 4 --n-delays 3 --rank-energy 0.99 --max-rank 50 \
  --dmd-regularization 1e-8 --analysis-seed 0 \
  --permutations 9999 --bootstrap-replicates 5000 --device cpu --workers 1 \
  --run-id hypotheses-subspace-20-seeds5-alltasks
```

Reuse selection, trained policies, ranking returns, and valid aligned episode
payloads. Refit all 100 policy systems per task and regenerate distances, H1,
Fig. 3B, bootstrap distributions, and figures. Manuscript H2 uses the same
returns and structural distances and should retain its numerical results.
The downloaded hypothesis summaries do not contain the missing original
trajectory payloads: obtain the complete source collections, checkpoint files,
and logs before running locally. Preserve the Walker2d mixed-training-regime
deviation; changing the estimator does not resolve it.

Identification runs in float64 (NumPy on CPU, Torch on requested CUDA). The
common rank still uses 99% delay-embedded **state** energy, capped at 50 and
common state support, rather than the subspace projection's energy spectrum.
Every projection must support that chosen rank numerically; unsupported ranks
fail rather than being lowered silently. `--n-delays 3` gives past/future
subspace horizons of three. Each episode must provide at least two windows;
all episodes must contribute, and no transition crosses a reset. Diagnostics
record projection support/spectra, episode windows/transitions, dtype, operator
dimensions and conditioning. These numerical checks do not establish held-out
identification accuracy on the real cohort.

For explicitly changed rank, horizons, regularization, or inference settings,
use `--exploratory-identification`; this does **not** allow ranking-episode reuse.
Choose settings using identification validation, never hypothesis p-values.
No failure changes the estimator or selects the custom subspace algorithm.
Do not reuse DMDc or old-schema distance caches. Use an unused run ID; to
reproduce DMDc with these inputs, specify `--backend dmdc` and a separate run ID.

InputDSA controllability scoring uses double precision and scales the SVD
cross-product to avoid overflow from large powers of fitted dynamics matrices.
Distances retain their original units, and fitted operators are unchanged.
Per-system spectral radii and scoring precision are recorded in analysis
diagnostics. Powers that exceed double-precision range fail with an explicit
error; changing rank or regularization requires an explicit analysis choice.

## Resuming training

Stop the old training process before resuming the same run directory. Add `--resume`
to the original command to skip completed policies and continue unfinished ones from
their latest checkpoint. You can change `--n-envs`, `--n-steps`, `--batch-size`, and
`--device`. Other training settings and the selection must match the original run;
`--training-steps` remains the total target per policy, not additional steps.

For example, try 32 environment workers while keeping 2,048 samples per PPO rollout:

```sh
uv run --locked python scripts/train_all_selected.py \
  --protocol corrected \
  --selected-csv artifacts/corrected/selection-20/selection/selected.csv \
  --run-id training-20-seeds5 --resume \
  --policy-seeds 0 1 2 3 4 \
  --tasks Hopper-v4 Walker2d-v4 Swimmer-v4 Ant-v4 HalfCheetah-v4 \
  --training-steps 500000 --device cpu \
  --n-envs 32 --n-steps 64 --batch-size 64
```

More workers can increase simulator throughput, but actual speed depends on CPU
capacity and subprocess overhead. Shorter per-environment rollouts change advantage
estimation even when total rollout size is unchanged. Policies still train sequentially.

New training runs save an atomic `.checkpoint.zip` approximately every 10,000
aggregate environment steps, after a completed PPO update (configure with
`--checkpoint-steps`). Resume restores policy weights, optimizer state, and timestep
counts, and continues TensorBoard logging. Simulator and random-generator states
restart, so continuation is not bit-for-bit identical. At most the work since the
last checkpoint is lost, and PPO may round the target up to a full rollout.

Runs started with the older code only saved completed policies: these are skipped,
but their unfinished policy must restart. Existing registry entries are preserved,
including policies outside the requested task/seed subset. Resume attempts are
recorded in `manifest.json`. Use `--resume`, without `--overwrite`.

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
  --protocol corrected --smoke --exploratory-reuse-ranking-episodes --tasks Swimmer-v4 \
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

The new [reservoir activity workflow](docs/WORKFLOWS.md#reservoir-activity-pca-and-cebra) plots PCA in 3D and two 2D projections, plus three separate CEBRA embeddings supervised by observations, executed actions, and rewards. It supports top/bottom K individual agents and separate or shared fits:

```sh
uv run --locked python scripts/plot_context_embeddings.py \
  --protocol legacy --group top --k 1 --run-id context-top1
```

This defaults to all five tasks and evaluates all discovered checkpoints when no saved episode returns are supplied. Add `--tasks Swimmer-v4 --limit 2 --smoke` for a small execution check; `--embedding-mode shared` pools selected agents within each task. Corrected model-registry and saved-data examples are in the workflow reference.

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
