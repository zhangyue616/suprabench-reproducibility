# Sources and third-party materials

The frozen scientific base was exported from a fixed SupraBench cold-split audit snapshot at commit `81b1d092b4d2bdd4a4f293f340e7d369395ac053`. Its [original attribution notice](source_notices/fixed_snapshot/THIRD_PARTY.md) and [original file-level assignments](source_notices/fixed_snapshot/FILE_LICENSES.tsv) are retained as source evidence.

The source notice identifies SupraBench and SupraBank, the data/code sources, changes and software dependencies. Its historical snapshot identity remains unchanged. [FILE_LICENSES.tsv](FILE_LICENSES.tsv) identifies the license and attribution applicable to each file in this new tree; the historical manifest does not license unrelated new files by implication.

The additional analyses consume five files from the fixed scientific snapshot through `additional_analyses/input_sources/SOURCE_BINDINGS.json`; `UPSTREAM_FILE_ASSIGNMENTS.json` retains their specific upstream assignments. Additional original software, documentation and data-derived results receive the explicit assignments in the current manifest.

No full manuscript, model checkpoint or third-party software package is redistributed. Dependency version pins identify an environment and do not change dependency licenses.


The current overview and numerical plotting resources add the contributors' retained figure sources and data-derived display values, with assignments in `FILE_LICENSES.tsv`. Plotting dependencies and any font substitution are documented in `figures/source/README.md` and `environments/figures.txt`. No third-party plotting library or font is bundled or relicensed.
