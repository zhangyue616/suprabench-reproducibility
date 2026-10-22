# Reproduction scope

## Saved-result verification

The main command uses frozen inputs and predictions. It does not fit models, construct new splits, or alter operational identity decisions.

| Group | Intended coverage |
|---|---|
| Canonical CPU | Original file bindings and exact saved repeat agreement; rescore all 325 block/pooled R² entries for 13 models and descriptor-RF six-null margins (pooled and double-cold block/equal-block/appearance-weighted), retaining dtypes and the 8 ULP tolerance |
| Graph models | Reaggregate all 350 saved jobs using their fixed record memberships and metric definitions |
| Tree models | Reaggregate all 420 saved jobs, retaining the saved prediction precision and the executed float32 handling for XGBoost |
| Operational-parent sensitivity | Replay the fixed-prediction analysis from frozen identity decisions, memberships and masks, with separate released-identifier and canonical-key references |

The software checks the pinned verification environment before these groups. Reports distinguish the environment precondition from the four scientific groups. Passing these checks establishes consistency of the supplied artifacts within this scope; it does not independently validate every analysis or regenerate every figure in the associated study.

The root `verify.py` wrapper also runs additional source/code/literal-target checks, rescores all 60 saved E1 jobs and 420 model metric rows with their guarded/standard margins and final contrasts, and inspects selected E2 outputs. E1 is an additional wrapper check; the core verifier still reports the four groups above. No new model fit or E2 decomposition rerun is part of this default command. See [REPRODUCING.md](../REPRODUCING.md) for individual commands and output restrictions.

Family diagnostics are supplied as descriptive saved results. They cover 184 model–seed–family records: 104 fixed-seed tree/null evaluations and 80 graph-model evaluations (two architectures, five seeds, eight families). The public verifier does not recompute these family diagnostics. Runs sharing one family holdout are not independent experimental replications.

## Additional saved-result audits

The materials below are further saved-result audits with their own input bindings and scope. The canonical CPU component of the model-versus-null aggregation is also checked within the canonical CPU group; the complete canonical GNN aggregation remains a separate command.

### Canonical GNN run and model-versus-null aggregation

The frozen primary run is in `results/canonical_gnn_run_a/`; the verification-only repeat is in `results/canonical_gnn_run_b/`. The primary run contains 12,964 test-appearance predictions and 50 metric rows. Its allow-listed provenance records the actual execution environment: Python 3.11.15, Torch 2.9.0+cu128, PyG 2.8.0, CUDA 12.8 and an NVIDIA GeForce RTX 5090. The scientific configuration and extracted training core are in `data/canonical_gnn/scientific_config.json` and `scripts/canonical_gnn/`.

The following command reaggregates the saved primary predictions against the same six canonical CPU nulls. It does not train a model:

```bash
python scripts/reaggregate_canonical_gnn.py \
  --gnn-predictions results/canonical_gnn_run_a/predictions_gnn.csv \
  --gnn-metrics results/canonical_gnn_run_a/metrics_gnn.tsv \
  --cpu-predictions results/canonical_cpu/run_a/predictions_cpu.csv \
  --cpu-metrics results/canonical_cpu/run_a/metrics_cpu.tsv \
  --frozen-targets data/canonical_cpu/features/tabular_features.npz \
  --output-dir runs/canonical_gnn_aggregation \
  --output-stem canonical_gnn_aggregation \
  --blocks-filename canonical_gnn_double_cold_blocks.csv
```

The frozen public results are `results/canonical_gnn_aggregation.json`, `.tsv`, and `results/canonical_gnn_double_cold_blocks.csv`. The equal-block double-cold margins are −0.2757452044 for GINE (2/12 positive blocks), −0.0235904965 for AttentiveFP (6/12), and +0.1198370303 for descriptor-RF (10/12). Appearance-weighted summaries are retained separately and are not substituted for equal-block values.

### Canonical GNN repeat comparison

`results/canonical_gnn_comparison/` retains the prespecified A/B repeat comparison as a 326-row table plus an allow-listed receipt. It reports 326 PASS, zero FAIL and zero symmetric-undefined comparisons under the frozen thresholds. The input predictions, exact historical comparator source and the narrow deprecation-warning compatibility wrapper are retained. This receipt is a repeat comparison; it is distinct from the model-versus-null reaggregation above. The exact historical comparator depends on its original canonical-build layout and is supplied for source audit rather than advertised as a standalone public rerun command.

### Diagnostic results

`results/diagnostics/` contains five saved JSON outputs exported by explicit scientific-key whitelists. Every key named `_meta` was omitted recursively; no value under such a key is used as evidence. The paired scripts under `scripts/diagnostics/` have repository-relative inputs and configurable outputs. This release did not execute them or fit their RF/GNN models. The marginal and leakage diagnostics require the upstream `records.parquet` with the SHA recorded in `DATA_SOURCES.md`.

The canonical RF specification is 300 estimators, `random_state=0`, `max_features="sqrt"`, and one job. The modern and diagnostic branches retain the same scientific RF parameters with `n_jobs=-1`. The modern run records scikit-learn 1.9.0. Attribution of the similarity-null run to that environment is inferred from its companion evidence because no per-run environment manifest was retained. Pregate and condition audits used scikit-learn 1.7.2. The saved identity-marginal double-cold fields report all four identity-keyed marginal branches as degenerate constant predictions; the global-mean control is constant by definition.

### Population, structural-neighborhood and recurrence audits

`scripts/build_structure_valid_subset.py` reconstructs the deterministic 2,609-to-2,383 filter from the SHA-bound upstream parquet into a fresh output directory. The saved manifest, exclusions and verification are under `data/population/`. `results/record_audit/` freezes the row/pair/task-identifier semantics, family residual decomposition and all 66 leave-two-block means.

`protocols/p1_d6sc_overlap/` keeps the outcome-blind Stage-A predicate and labels separate from Stage-B fixed-prediction scoring. `results/strict_recurrence_sensitivity/` and `scripts/reaggregate_37_mask.py` retain the exact 37-appearance mask and the seven-model before/after block scores. The retained-population audit reuses saved predictions across the same 12 blocks; it does not remove training exposure or refit any model.

`scripts/parent_identity_b1/`, `protocols/parent_identity_b1/` and `results/parent_identity_b1/` retain the five v1 identity layers, parent construction, frozen membership masks and Stage-C fixed-prediction results. Historical execution logs are outside the public scientific package.

## Scoring conventions

Scored record populations, identifier schemes and aggregation rules are part of each result's definition. A pooled held-out score and an equal-block double-cold margin answer different questions and must not be substituted for one another. Null comparisons use the same scored populations as the corresponding model comparison.

Full-precision reference comparisons use an eight-ULP rule where specified. Tree R² reaggregation retains its historical strict absolute tolerance of `1e-12`. Exact bindings, identifiers, statuses and memberships are checked exactly. Rounded producer references retain their original precision and are not promoted to full-precision evidence.

Undefined scores remain undefined. The operational-parent replay retains its guards for fewer than two observations, constant targets, near-constant predictions (standard deviation at most `1e-9`, population convention) and nonfinite values. It does not replace undefined scores by zero. Historical target and prediction dtypes are retained when they affect scoring.

## Operational-parent sensitivity

The analysis scores retained test appearances with already fitted predictions. The frozen representation rules and group decisions are operational definitions, not adjudications of experimental chemical equivalence. The primary scenario retains strict identities for uncertain groups; alternative scenarios use their separately frozen masks.

The released-identifier and canonical-key evaluations have different memberships and references. Test appearances, unique physical rows and distinct identities are different counting units. An appearance can be retained in one layer or scenario and removed in another. The results do not estimate performance after parent-disjoint retraining, and alternative scenarios are not statistical upper and lower bounds.

## New training

Training commands are separate from saved-result verification. Released tree/graph commands use fresh output directories under `runs/`; the additional E1 `run-e1` command requires a fresh absolute directory outside the repository and companion. They retain the frozen jobs, memberships and settings. They are protocol-compatible training entrypoints, not a guarantee of bitwise recovery of historical models. E1 runs all twelve canonical double-cold descriptor-RF/six-null baseline blocks before its purge arms, with the original 8 ULP baseline stopping rule; no E1 training was performed for this release. Apart from the E1 baseline arm (descriptor-RF and its six nulls on the twelve canonical double-cold blocks), no canonical CPU retraining entrypoint is supplied: none for the canonical host-cold or guest-cold splits, and none for the other six canonical tree configurations on any split.

The original released-job graph preparation refers to an unavailable pilot preparation-script hash; one historical job used that pilot runner. GPU model, deterministic algorithms and TF32 were not all fixed for that released-job history. The later canonical GNN run has a separate hard-determinism contract and actual Torch 2.9 runtime record. The frozen tree feature array is supplied because its original 3D feature preparation used a wall-clock cutoff. The public training commands do not regenerate that feature array.

Use the README commands and the corresponding pinned requirements. Full graph training requires a compatible CUDA environment. A full suite is substantially more expensive than saved-result verification; start with one complete specified job when testing an installation.
