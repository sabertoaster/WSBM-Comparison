# Validation of the research refactor

Run the first-party checks with:

```sh
uv run --locked pytest -q
uv run --locked ruff check src scripts er_mrl tests main.py
uv run --locked ruff format --check src scripts er_mrl tests main.py
```

The suite covers CLI help/import safety, protocol validation, paths outside the checkout, write-free dry runs, selection fingerprints and original identifiers, immutable matrix loading, weighted clustering, exact expected density, action encoding, resets, five task velocity masks, independent episode embeddings, ragged-episode DMDc, permutation statistics, event completeness, and timestep alignment.

The [README smoke workflow](../README.md#small-end-to-end-check) validates the complete corrected path on a small Swimmer experiment. Full training and publication-scale permutation analyses are deliberately separate. No test deletes project outputs or modifies DSA.

Test artifacts are created in pytest temporary directories. The manual validation uses `/tmp/wsbm-validation`; its manifests retain failed attempts and subsequent successful runs separately, so failures remain inspectable.

## Verification results

Verified on 2026-10-01 with the existing Python 3.10 virtual environment:

| Check | Result |
| --- | --- |
| First-party pytest suite | 63 passed |
| Ruff lint and formatting | Passed for all 44 first-party Python files |
| Locked dependency metadata | `uv lock --check --offline` passed |
| Wheel build | Hatchling produced `wsbm_comparison-0.1.0-py3-none-any.whl` |
| Corrected selection | Five selected 20-neuron networks; two sampling workers; saved matrices match descriptor hashes |
| Corrected training and collection | Four reservoirs plus PPO baseline; 64 steps each; one deterministic 100-step Swimmer episode per model |
| Corrected paper analyses | Five H1 metric tests, global H2, pair tables, operator diagnostics, and figures exported |
| Fixed-structure policy replicates | Two policy seeds, two environment workers, 32 steps; identical W and input weights across seeds/workers; two evaluation episodes per seed |
| Synthetic OU analysis | Two reservoirs, 100 samples, rank 1; corrected DMDc and historical custom-subspace routing both completed |
| Legacy compatibility | Fresh direct training, motif plotting, and loading/evaluating an existing 500,000-step Swimmer checkpoint completed |
| Ranking and descriptor PCA | Corrected top/bottom IDs and performance-colored PCA exported from the collected returns |

Final manual run names are `corrected/final-selection`, `corrected/final-training-v2`, `corrected/final-collection-v2`, `corrected/final-analysis-v2`, `corrected/replicate-check-v2`, `corrected/replicate-collection`, `corrected/synthetic-check`, `legacy/legacy-synthetic-check`, `legacy/final-legacy-training`, `legacy/final-figure`, and `legacy/historical-checkpoint`. They are validation artifacts outside the repository and are not publication results.

The tiny selection supplies fewer than three reservoirs per stratum, so within-stratum H2 is explicitly undefined. That is expected for this smoke check. Full hypothesis conclusions require the configured 100-reservoir experiment and sufficient policy training.

The suite emits expected external DSA warnings about separate state comparisons and Gymnasium deprecation notices for the retained paper `v4` IDs. DSA and subprocess imports can also emit TensorFlow runtime notices. The two-worker training check required execution outside the filesystem sandbox because Python's forkserver socket was blocked; its initial failed run was retained separately.

No files were deleted. Existing selected CSVs, figures, checkpoints, trajectories, the DSA Git link, and external DSA source were preserved.
