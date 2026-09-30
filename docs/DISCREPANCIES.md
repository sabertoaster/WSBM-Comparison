# Paper and code discrepancy register

This register compares the supplied 19-page manuscript with the repository inspected before the refactor. It identifies why historical outputs must remain distinct from corrected experiments. It does not certify the paper's numerical results or reconstruct missing historical execution settings.

## Protocol differences

| Issue | Historical code | Corrected protocol | Consequence |
| --- | --- | --- | --- |
| Reservoir count versus size | Five strata × 20 CSV rows; selection declares 200 neurons, RL defaults to 100 | Select and train the same persisted network; full preset uses 100 configurations of 200 neurons each | Existing training results do not establish behavior of the selected 200-neuron networks. Reselect/retrain for corrected claims. |
| Selection identity | Root and `scripts/selection/selected.csv` each have 100 rows, with only 24 shared reservoir seeds | Require explicit corrected selection and record its SHA256; retain both historical CSVs | Choose the actual historical dataset explicitly. Neither CSV substitutes for the other. |
| Weight contrast and dispersion | Selection uses `(centre ± contrast) × 0.1` and a block dispersion matrix; RL uses `centre ± contrast/2` and scalar `sigma_ratio × centre` | One canonical construction plus saved realized W | Old structural descriptors and the trained networks can describe different matrices. |
| Topology and community sizes | Selected density ratio, size heterogeneity, core fraction, and mixing fraction are lost during RL reconstruction | Keep actual assignments and block probability matrices | Rerun selection and training to test the intended structural manipulations. |
| Negative weights | RL converts measured `frac_negative` into an additional forced-negative probability | Negative weights arise from the sampled Gaussian distributions; no additional sign flip | Old realized block means differ from the intended prior. |
| Density control | Equal-block approximation plus clipping, even for unequal sizes/core topology | Solve expected edge count over actual unordered node pairs; reject infeasible draws | Corrected pools and MaxMin picks can change. |
| Clustering descriptor | Comment says weighted Onnela; calculation uses binary adjacency transitivity | Actual Onnela weighted clustering on normalized absolute weights | Regenerate descriptors and structural selection. |
| Observation protocol | `--del_obs` defaults False; velocity masking lacks the five locomotion tasks | Verified masks for all five tasks, applied to PPO too | Existing full-observation checkpoints cannot represent corrected partial-observation agents. |
| Action inputs | Reservoir wrapper feeds zeros instead of the executed action | Encode the actual preceding action and reward alongside the next observation | Retrain corrected policies; changing only evaluation would invalidate checkpoint inputs. |
| Episode state | Reservoir state persists across resets by default | Reset state per episode | Corrected evaluation no longer carries hidden memory between episodes. |
| Replicate seeds | Reservoir and optimization seed are coupled; first-per-stratum replicates change W | Keep W and read-in seed fixed, vary policy seeds separately | Corrected replicates can estimate optimization variation for a fixed reservoir. |
| DMDc backend | `InputDSA` defaults to `SubspaceDMDc`; a `dmdc` backend string follows its custom-subspace branch | Pass the explicit external `DMDc` class | The historical label does not prove standard DMDc was used. Corrected matrices require recomputation. |
| Dynamics defaults | Several scripts use N4SID, although the main text discusses standard DMDc as the stable choice | Explicit backend in arguments and metadata; no fallback | Numerical settings must accompany every figure. |
| State metric | Separate state comparison inherits external Wasserstein defaults | Preserve that default but expose and record metric/optimization settings | Joint-aligned state distance and separate state distance are distinct computations. |
| Trajectory alignment | Tracker records reset, step, and vector autoreset calls together; input is paired with post-input state | Save per-episode transitions with the forcing that advances the recorded state | Do not interpret legacy arrays as correctly aligned multitrial dynamics without boundary metadata. |
| H1/H2 names | Historical H1 scripts plot structural versus state distance; historical H2 plots DSA matrices | Preserve filenames and document them; add a paper-hypotheses entry point | Filename suffixes do not identify the paper's actual hypothesis tests. |
| H2 implementation | Existing scripts do not supply the complete deterministic-return pairwise analysis in the paper | Evaluate returns, average policy replicates equally, export global/within-stratum tests | Published H2 numbers are not reproduced by a documentation refactor. |
| Statistical dependence | Reservoir pairs share nodes in the distance matrices | Permute reservoir labels, not individual pair entries | Permutation p-values need not equal naive pairwise Spearman p-values. |
| Logs and paths | First event file/seed selected in some readers; several relative paths and `uv run` subprocess arguments are broken | Read complete events deterministically, align by timestep, use isolated run registries | New log summaries can differ from incomplete historical summaries. |

Legacy mode preserves the historical reservoir conversion, zero-action encoding, state persistence defaults, synthetic constructors, and DSA backend routing. It also fixes orchestration and reporting bugs: paths, safe imports, complete log loading, stable identity, and run isolation. Therefore legacy mode is a compatibility tool, not a guarantee of byte-identical reruns of every figure. Existing arrays, figures, and checkpoints are preserved in place.

## Interpretation limits in the manuscript

The manuscript's Section 5 begins with SubspaceDMDc terminology and later states that standard DMDc was chosen. The external class/backend distinction above must be resolved when describing the next experiments.

The Introduction describes connectome-derived priors, but these scripts sample synthetic WSBM motifs; they do not fit block parameters to empirical connectome data. Describe these experiments as biologically motivated generative priors unless an empirical fitting pipeline and data are added.

The RC background discusses a linear readout and readout regularization. The locomotion pipeline trains PPO's actor–critic MLP on context vectors, with default SB3 policy architecture. It does not fit a ridge readout. Report the actual policy architecture and PPO settings rather than implying a ridge parameter was manipulated or held fixed in this RL experiment.

The paper reports one policy run per reservoir and correctly identifies optimization stochasticity as a limitation. The corrected preset enables five independent policy seeds per fixed reservoir. Evaluation episodes add rollout variation, not independent reservoir samples. A short smoke run cannot validate dynamical convergence, the reported return ranges, or the published correlations.

The main text refers to input-operator singular spectra, but the historical analysis scripts do not export a complete spectrum-based hypothesis test. The new diagnostics expose operator norms/conditioning and input spectra for inspection; any new significance claim needs a specified analysis and full experiments.

## Preserved external code

ReservoirPy is the user's fork at `wsbm-fix`, locked to `eae1746f50bd5a798bb964471f273dbb03260ad1`. DSA is the local external checkout at `7ee20ff33c03d37e4f7bb869dbb1a0588637427d`. It is tracked as a Git link without a `.gitmodules` mapping. This refactor neither changes that link nor rewrites DSA code or Git objects. Its distribution metadata and Python `__version__` differ; use the commit and installation metadata for provenance.

The ER-MRL-derived helpers are retained under `src/er_mrl` with attribution to [the upstream project](https://github.com/corentinlger/ER-MRL). Legacy Optuna helpers remain available but are never called by the developmental experiment pipeline. The local `src/wsbm_esn.py` reference models remain available; the corrected experiment factory uses the ReservoirPy fork and persisted matrices.
