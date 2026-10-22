# E1/E2 public location adapter

From the repository root, using Python 3.11.15 and the recorded environment:

```text
python -B additional_analyses/portable/reproduce.py check
python -B additional_analyses/portable/reproduce.py inspect-saved
python -B additional_analyses/portable/reproduce.py rescore-e1 --output-dir [new-absolute-rescore-output]
```

The repository root contains the fixed snapshot `data/` and `results/` trees.
`additional_analyses/` is the companion root, and its `original_run/` directory
contains the exported historical contract, inputs, masks and saved E1/E2 outputs.
The snapshot root defaults to the checkout root; `--snapshot-root .` is also valid
from that root. The source registry uses `scope` plus `relative_path`: snapshot
sources are relative to the snapshot root, and original-run sources to
`additional_analyses/original_run/`. The public contract's `experiment_root` is
`.` relative to `original_run/`; it is resolved by this adapter, not by the cwd.
Other relative paths in saved provenance are relative to the repository root.

The public export changes only location/provenance metadata, label, comment, docstring and message wording,
the corresponding byte bindings, and this adapter. `PUBLIC_EXPORT_MANIFEST.json` in the companion
root records original archive-member identities, exported identities and field
changes. It never republishes private workstation paths. `stage_a_freeze.json`
retains historical Stage A facts but binds the exported contract and scripts;
it is not an unmodified original receipt or evidence of a new Stage A run.

`check` verifies the public executable/contract/adapter bindings and all eight
unchanged scientific source identities. It compares canonical CSV target strings
with the stored feature target strings using exact text equality. No scores are
computed and no estimator is fitted. `inspect-saved` runs the same checks, then
accesses two rows in each of six saved CSVs and saved completion/summary records.
Each invocation hashes bound files once through an in-process cache. A caller can
run check and inspection on one Adapter instance to reuse identities.

`rescore-e1` runs the same bindings/literal-target checks and the strict repository verification environment check. It uses the preserved E1 scoring and summary functions to recompute all 60 saved jobs, 420 model metric rows, guarded/standard block margins, twelve-block arm means and the parent-minus-random/baseline contrasts. Values are compared within 8 ULP; identities, statuses and undefinedness are exact. No estimator is constructed or fitted. The fresh absolute output may be under the repository's ignored `runs/` directory or outside the preserved inputs. The default `verify.py` calls this rescore as well as inspection; it does not call `run-e1` or `run-e2`.

The E1/E2 scientific runner code is unchanged apart from comments and messages. The adapter supplies its explicit
`staged_io` replacement and maps source/output locations. Model formulas, six
nulls, masks, seeds, dtype handling and the 8 ULP baseline check are retained.
The five locked thread environment variables are set for `run-e1` before the
first numerical-library import, before input checks, and before any fit.

An explicitly requested replay has separate entry points:

```text
python -B additional_analyses/portable/reproduce.py run-e2 --output-dir [new-absolute-e2-output]
python -B additional_analyses/portable/reproduce.py run-e1 --output-dir [new-absolute-e1-output]
```

E2 recomputes a rank diagnostic without fitting. E1 performs 60 jobs / 300 RF
fits. Neither command is part of saved-output inspection or public export.
New outputs must be outside the repository and external snapshot, in an explicit
absolute directory that does not exist. No automatic retry or resume is supplied.

`run-e1` executes all twelve canonical double-cold baseline blocks before the intervention arms, with the frozen canonical feature arrays/memberships, RF recipe (300 trees, `random_state=0`, `max_features=sqrt`, `n_jobs=1`), the same six nulls and original dtype handling. The baseline prediction and MAE/R²/Spearman comparisons stop on differences above `8*ulp(reference)`; keys and statuses must agree exactly, with no tolerance widening. It is a full five-arm training command, not a baseline-only selector. This release did not run it. No other canonical CPU retraining entrypoint is supplied: none for the canonical host-cold or guest-cold splits, and none for the other six canonical tree configurations on any split.

Historical Stage A construction scripts are retained for source inspection;
their local validation/projection receipts and the source archive are not
redistributed. Run them only with the original separately retained construction
inputs. Two `canonical_formula_sources` are historical static source references,
not imports or dependencies of the current portable runners; their original
bytes/SHA are retained and their paths explicitly say not redistributed.
The environment versions in the contract are historical observations. The
repository does not install dependencies or assert that another environment
reconstructs the historical fitted models byte for byte.
