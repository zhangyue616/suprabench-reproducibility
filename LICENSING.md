# File-level licensing

No single license applies to every file. [FILE_LICENSES.tsv](FILE_LICENSES.tsv) is the current assignment for every public file, including documentation and license texts.

## Retained scientific snapshot

Files derived from the fixed scientific snapshot retain the exact license identifier and attribution assigned by its original file manifest. The snapshot is identified in [SOURCE_IDENTITIES.md](SOURCE_IDENTITIES.md). Original notices are retained in `source_notices/fixed_snapshot/`; their historical paths and file counts describe that source snapshot, not the expanded current tree.

The original software expressly assigned MIT remains MIT. Benchmark data and data-derived scientific artifacts assigned CC BY 4.0 retain attribution to SupraBench and, where applicable, SupraBank, along with their source identities, transformation descriptions and the CC BY 4.0 link. No upstream endorsement is implied.

## Additional analyses and public documentation

For this public release, the contributors license their newly supplied original software, path adapter, protocol text and associated documentation under MIT, as individually listed. Scientific inputs, identity/decision tables, derived masks, saved predictions, scientific results and computational-record metadata are assigned CC BY 4.0 with the applicable benchmark attribution. These are explicit assignments for this release. License fields recorded inside retained source records describe those sources; the per-file assignments in FILE_LICENSES.tsv govern this release.

The grants apply to rights held by the contributors. They do not grant rights in third-party software or material that the contributors do not control. Original source-specific obligations remain applicable. No full manuscript or third-party paper text is licensed or distributed through this repository.

Copyright (c) 2026 Contributors to the SupraBench reproducibility study.

Full license texts: [MIT](LICENSES/MIT.txt) and [CC BY 4.0](LICENSES/CC-BY-4.0.txt). `LICENSE-TEXT` entries identify the supplied license documents; those texts are not relicensed as project software.

## Attribution and modifications

Scientific data originate from [SupraBench](https://github.com/Tianyi-Billy-Ma/SupraBench), with applicable binding records attributed to [SupraBank](https://suprabank.org/). Filtering, condition bucketing, identity mapping, split construction, prediction generation and aggregation are described in `docs/DATA_SOURCES.md` and the corresponding protocols. The additional analyses use those inputs for operational-parent training purges and saved-prediction rank decomposition.

Public-export modifications are listed in [PUBLIC_EXPORT_MANIFEST.tsv](PUBLIC_EXPORT_MANIFEST.tsv). Path and provenance metadata, comment, message and label wording in exported scripts and records, the public adapter's bindings and documentation may differ from their source archives. Scientific parameters, memberships, predictions and result values are preserved. The additional public binding record is explicit about its source and is not a new Stage-A scientific freeze.

Dependencies are not vendored or relicensed. NumPy, pandas, SciPy, scikit-learn, RDKit, XGBoost, LightGBM, PyTorch and PyTorch Geometric remain governed by their respective upstream terms. See [THIRD_PARTY.md](THIRD_PARTY.md).


## Figure overview and portable rendering resources

The current Figure 1 PNG and PDF assets, the historical Figure 1 SVG, frozen numerical display inputs and display recipes are scientific material assigned CC BY 4.0, with the study and applicable SupraBench/SupraBank attribution. Newly supplied or adapted plotting code, environment specifications and documentation are assigned MIT to the extent of the contributors' rights. The public manifest records unchanged source payloads separately from portable code or projected recipe fields. These assignments do not license dependencies or fonts, and do not add a complete manuscript or publication-asset bundle.
