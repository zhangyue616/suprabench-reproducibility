# Frozen figure display inputs

These files are display inputs, not newly calculated scientific results.
Figure 3 uses 45 stored points and nine stored focal comparisons. The six CSVs
provide Figures 4, 5, 6, S1, S2 and S3. Three small supporting recipes supply
only Figure 4 positive-block labels, Figure 6 family counts, and S3 stored mean
ticks/historical aggregates. No other historical recipe metadata is needed.

S1/S2/S3 CSVs retain their original bytes. Figures 4/5/6 preserve every
scientific cell string, column and row order; only source labels are mapped to
public scientific paths and the CSVs are serialized as UTF-8/LF. Figure 3's
stored value fields are retained while its source labels use public aliases.

| Display | Scientific source |
| --- | --- |
| 3 | results/canonical_cpu/ and results/canonical_gnn_aggregation.tsv |
| 4 | data/reference/suprabench_baselines_modern_results.json, results/tree_saved/ and results/gnn_saved/ |
| 5 | results/diagnostics/ (separate similarity, condition/source and pregate branches) |
| 6 | results/family_diagnostics/family_metrics.csv |
| S1 | data/population/ and results/diagnostics/suprabench_cond_audit_results.json |
| S2 | results/diagnostics/suprabench_sim_null_results.json and data/reference/suprabench_baselines_modern_results.json |
| S3 | results/gnn_saved/; historical reference aggregates are not recovered historical predictions |

The renderer does not recompute predictions, fitted models or scientific
aggregations. A display coordinate, mean tick or historical reference remains
the frozen display quantity it originally represented. See
[figure source instructions](../../figures/source/README.md).
