# Source identities and public-export changes

The scientific base is a fixed SupraBench cold-split audit snapshot, source commit `81b1d092b4d2bdd4a4f293f340e7d369395ac053`, which is not redistributed separately; files retained from it are identified by their recorded SHA-256 and byte counts. This repository has its own history and does not modify that snapshot.

| Source | Current location | Treatment |
|---|---|---|
| Fixed scientific snapshot | Original scientific `data/`, `results/`, `scripts/`, `protocols/`, `environments/`, `docs/` and license texts | Frozen data, folds, features, predictions and numeric result tables retain their byte identities and per-file licenses; wording edits to some configuration, record, script and documentation files and public verification additions are recorded in the current manifest. The current overview and plotting sources are described below |
| 10 October 2026 computational companion | `additional_analyses/` | Selected scientific inputs, code and saved results; local path metadata removed or mapped; public bindings identify modified metadata/scripts |
| 27 September 2026 parent-identity group inventory | `results/parent_identity_b1/identity/v2_group_review_inventory.csv` | Original 47-group member inventory; the scientific decision table is retained in the same directory |

[PUBLIC_EXPORT_MANIFEST.tsv](PUBLIC_EXPORT_MANIFEST.tsv) maps current files to source archive identifiers/members and source/public byte identities. Source archive identifiers are provenance labels; original private archives are not redistributed. For retained figure-production, figure-asset and parent-identity sources, archive identifiers and member paths are aliases; the recorded SHA-256 and byte count are the binding identities. Newly written public files are labeled as such. The manifest excludes its own digest to avoid self-reference; Git binds the final manifest with the rest of the public tree.

The additional contract/script binding manifest describes a public export, not a rerun or modification of the historical experiment. Unchanged scientific inputs, masks, prediction tables and results retain their original byte identities. Changed fields and text are recorded separately; no changed file is claimed to be byte-for-byte identical to its source.

The original notices under `source_notices/fixed_snapshot/` describe the older snapshot. The current README, file-license table and reproduction guide govern navigation and scope for this repository. Older static references to nonredistributed historical source files remain identified as such; they are not prerequisites silently fetched from private locations.


## Current overview and plotting resources

The current `figures/Figure_1.png` and `figures/Figure_1.pdf` are the overview assets, copied without changes; `figures/source/Figure_1.svg` is an earlier historical schematic and is not the source of the current Figure 1. The PNG and PDF supersede the older display assets for navigation without changing scientific results.

The portable numerical renderer uses selected retained plotting functions and the necessary frozen display values for Figures 3–6 and S1–S3; it draws them in an earlier plotting layout, not the paper's panel arrangement. Public input projections remove local paths and unused fields; scientific display values and layout settings remain identified in the rendering-source map. The public manifest distinguishes original payloads, selected-field projections and code-portability edits. Rendering these values is separate from reaggregating saved predictions or reconstructing missing historical training runs. Figure 2's complete composition chain is not represented as a portable data-driven generator.
