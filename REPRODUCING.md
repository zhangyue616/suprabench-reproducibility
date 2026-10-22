# Reproducing the supplied results

The default workflow checks saved artifacts. New model fitting is separate and is not needed for this check.

## Environment and one-command check

Use Python 3.11.15 with `environments/verify.txt`: NumPy 1.26.4, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.9.0 and RDKit 2026.03.3. Install these in a separate environment, then run:

```bash
python verify.py
```

The command creates a fresh subdirectory under this repository’s `runs/`, following the preserved verifier’s output protection. An optional `--output` must name a new subdirectory there. The command can be invoked from another working directory by providing the full path to `verify.py`; both inputs and the default output are resolved from its repository. Existing scientific inputs and results are never overwritten. The `runs/` directory is ignored by Git.

The check was run on a separate copy of the tracked files with a different working directory. Its interpreter, platform, commands expressed with portable path placeholders, outcomes and limits are in [validation/SAVED_RESULTS_VERIFICATION.json](validation/SAVED_RESULTS_VERIFICATION.json). The receipt does not retroactively verify missing historical runs.

### Platform coverage

The recorded run used Windows x86-64. Linux and macOS have not been tested. The environment check requires the exact pinned versions, including Python 3.11.15, and has no relaxation switch. The 8-ULP numeric rule is strict for R² values, margins and contrasts close to zero: there, a one-ULP difference in an underlying error ratio or mean can exceed the tolerance, so a different CPU, math library or summation order may stop the check without any scientific difference.

## Coverage matrix

| Material | Default check | Limit |
|---|---|---|
| Canonical CPU | Original bindings and saved A/B agreement; recompute all 325 block/pooled R² entries for 13 models, plus descriptor-RF pooled and double-cold block margins against the same six nulls | Saved predictions only; exact memberships/definedness and 8 ULP numeric comparison; no new fit |
| Released GNN results | Reaggregate 350 saved jobs with their fixed memberships and scoring definitions | Does not recreate the unavailable pilot preparation source, checkpoints or training trajectories |
| Released tree results | Reaggregate 420 saved jobs with the retained prediction precision and historical XGBoost float32 handling | Uses supplied frozen features; does not recreate historical wall-clock-limited feature preparation |
| Parent-identity fixed-prediction sensitivity | Replay the retained analysis using frozen identities, memberships and masks | Operational rules, not expert chemical adjudications; rules were built from structure/name inputs without folds, affinities or predictions, but earlier results were already known, so no blinding is claimed |
| Additional E1/E2 | Validate eight source bindings, public contract/script bindings and literal targets; rescore all 60 saved E1 jobs/420 model metrics, guarded and standard margins, arm means and contrasts; inspect selected E2 outputs | No E1 fit or E2 decomposition rerun |
| Canonical GNN aggregation and A/B comparison; family, population, structural-neighborhood and recurrence audits | Supplied with their own definitions and saved results | Outside the default four-group recomputation; use the specific scope in `docs/REPRODUCTION.md` and `docs/SCIENCE_COVERAGE.md` |

The 184 family records are descriptive saved results and are not recomputed by the default verifier. Legacy early GNN evidence is partly aggregate-only; later saved jobs do not fill that gap. Undefined values, scoring populations, metric guards and aggregation order retain their original meanings.

## Individual entrypoints

These are the commands called by the combined wrapper:

```bash
python -B scripts/verify_saved_results.py --root . --output runs/core-verification
python -B additional_analyses/portable/reproduce.py inspect-saved --snapshot-root .
python -B additional_analyses/portable/reproduce.py rescore-e1 --snapshot-root . --output-dir [new-absolute-output-directory]
```

The additional entrypoint also offers `check` for bindings alone. See [additional_analyses/README.md](additional_analyses/README.md) for the public path mapping and saved-result scope.

To run only the canonical CPU binding and rescore checks, use `python -B scripts/rescore_canonical_cpu.py --root . --output runs/cpu-rescore` with a fresh output directory. The rescore retains float32 targets, float32 block predictions for XGBoost, float64 block predictions for other CPU models and float64 pooled accumulation. It preserves the constant-prediction guard and selects the largest finite R² among the same six nulls separately within each block. The report includes actual/stored R² comparisons, the three pooled descriptor-RF margins, twelve double-cold margins and their equal-block and appearance-weighted means. A pooled margin is a separate quantity from the mean of block margins.

## Separate training and other analyses

The frozen scientific scripts and training entrypoints are retained for inspection and explicitly requested protocol-compatible runs. Training has separate dependencies and a materially different cost from saved-result checks. Graph training requires its corresponding CUDA/PyTorch/PyG environment. The released-job environment and the canonical GNN historical runtime differ; do not substitute one for the other. Exact historical recovery is not guaranteed.

E1's `run-e1` would fit 300 RF models across 12 blocks and five arms. E2's `run-e2` computes a rank decomposition from saved predictions and makes no new fit. Neither is called by the default public verification command. New outputs require a fresh directory; retained results are never overwritten.

The existing E1 baseline arm is also the public protocol-compatible training path for canonical double-cold descriptor RF and its six nulls:

```bash
python -B additional_analyses/portable/reproduce.py run-e1 --snapshot-root . --output-dir [new-absolute-output-directory-outside-repository]
```

This command runs all twelve baseline blocks before any purge arm. It uses the frozen canonical features/masks, 300-tree RFs (`random_state=0`, `max_features=sqrt`, `n_jobs=1`), additive host/guest prediction minus the own-arm training mean, and similarity-weighted k=5. Five thread variables are locked to 1 before numerical imports. Baseline prediction values and MAE/R²/Spearman must match the saved canonical values within `8*ulp(reference)` with exact keys/statuses; a mismatch stops the run without tolerance widening or automatic retry. The command runs the complete five-arm E1, rather than a baseline-only selection. No new E1 training was performed for this release; its saved baseline comparisons are historical evidence. Apart from this baseline arm, no canonical CPU retraining entrypoint is supplied: there is none for the canonical host-cold or guest-cold splits, and none for the other six canonical tree configurations (ECFP_RF, 3D_pair_RF, XGB_pair, XGB_ecfp, LGBM_pair and LGBM_ecfp) on any split, whose saved predictions are rescored only. The later released tree/GNN training adapters do not fill that canonical-protocol gap.

The upstream `records.parquet` is not redistributed and is not required for the default saved-result check. Tasks that reconstruct the raw-record subset or run diagnostics that need it must obtain the bound file using [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

Early per-epoch logs, checkpoints and some historical dependencies are not supplied.


## Rendering figures from retained sources

Figure rendering is separate from `verify.py`. The Figure 1 overview is supplied as [PNG](figures/Figure_1.png) and [vector PDF](figures/Figure_1.pdf) assets, which `scripts/render_paper_figures.py` exports directly (`asset_export`) without reconstructing the schematic from raw molecular data; an earlier schematic remains archived at [`figures/source/Figure_1.svg`](figures/source/Figure_1.svg) as a historical file. Numerical Figures 3–6 and S1–S3 use frozen display inputs and adapted plotting functions; they are drawn in an earlier plotting layout, and the panel arrangement, dimensions and annotations of the paper's figures are not generated by this entrypoint. See [figures/source/README.md](figures/source/README.md) and [environments/figures.txt](environments/figures.txt); each drawing run writes its own rendering receipt to the output directory. No new model fit or scientific aggregation is part of this drawing command. Outputs go to a fresh ignored directory; the upstream Python layout and composition chains for Figures 1 and 2 are not claimed to be portable here.
