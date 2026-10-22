# Additional analyses: saved E1 and E2 diagnostics

This directory supplies the existing 10 October 2026 analyses: fixed-test operational-parent training purge (E1) and decomposition of saved cyclodextrin rank correlations (E2).

- [Portable entry point and commands](portable/README.md): inspect saved outputs, or explicitly request a new protocol-compatible run.
- [Scientific contract](original_run/analysis_contract.json), [frozen membership masks](original_run/masks/) and [saved results](original_run/results/).
- [Public export differences](PUBLIC_EXPORT_MANIFEST.json): original file identities, changed fields and public byte bindings. Only paths, metadata, comment, docstring, message and label wording, and the portable location adapter were changed; scientific settings, source identities, runner calculations and saved numerical outputs were preserved.
- [Historical execution summary](EXECUTION_RECORDS.json): selected invocation, exit and completion records, with workstation metadata removed.

The default repository command verifies the core saved results, checks the additional source/code bindings, rescores all 60 saved E1 block-arm jobs/420 model metrics and their guarded/standard margins, equal-block arm means and contrasts, and accesses selected E2 saved outputs. It does not refit E1 or rerun E2. The original E1 record reports 60 completed block-arm jobs and 300 completed fits; these are historical completion records. No model fitting was performed for this release. `rescore-e1` uses the unchanged historical scoring functions and an 8 ULP tolerance with exact identities, statuses and literal targets.

The public `stage_a_freeze.json` binds the exported contract and adapter. Its manifest identifies changed historical metadata and omitted construction-only evidence; the export did not repeat Stage A or refreeze scientific classifications. Two historical static formula-source identities remain recorded without redistributing those old files. The actual retained E1/E2 runners do not import them. Historical constructors are supplied for source inspection, not as the public reproduction entry point.

The unchanged scientific files retain their original identities. Public path, metadata and wording transformations are explicitly documented and are not claimed to be original historical bytes. See the root [licensing terms](../LICENSING.md) and [per-file license manifest](../FILE_LICENSES.tsv).
