"""Relocated source checks and explicitly requested execution of preserved E1/E2.

Default check/inspect commands do not fit estimators or recompute scores.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import types

from staged_io_adapter import Adapter, BindingError, write_json

MODELS = ("PAIR_RF", "sim_knn_k5", "additive", "host_only_RF", "guest_only_RF", "cond_only_RF", "source_only_RF")


def require_literal_equal(canonical: str, feature: str, label: str) -> None:
    if canonical != feature:
        raise BindingError("literal target text mismatch: " + label)


def literal_target_check(adapter: Adapter) -> dict:
    """Compare saved canonical target strings to stored feature strings, not floats."""
    import numpy as np

    feature_path = adapter.source_path("canonical_features")
    prediction_path = adapter.source_path("canonical_baseline_predictions")
    for sid, path in (("canonical_features", feature_path), ("canonical_baseline_predictions", prediction_path)):
        adapter.verify_source(adapter.contract, sid, path)
    with np.load(feature_path, allow_pickle=False) as loaded:
        rows = loaded["row_index"]
        target_text = loaded["y_true_serialized"]
    if rows.tolist() != list(range(len(rows))) or target_text.shape != rows.shape or target_text.dtype.kind != "U":
        raise BindingError("stored literal target/row authority has an unexpected shape or dtype")
    expected_groups = {(f"double_cold/seed_{fold}", model) for fold in range(adapter.contract["e1"]["n_blocks"]) for model in MODELS}
    keys, groups = set(), set()
    with prediction_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"split", "block_id", "model", "row_index", "y_true"}.issubset(reader.fieldnames or ()):
            raise BindingError("canonical target/key fields are missing")
        for row in reader:
            if row["split"] != "double_cold" or row["model"] not in MODELS:
                continue
            index = int(row["row_index"])
            key = (row["block_id"], row["model"], row["row_index"])
            if key in keys or not 0 <= index < len(rows):
                raise BindingError("duplicate or invalid canonical target key")
            require_literal_equal(row["y_true"], str(target_text[index]), "/".join(key))
            keys.add(key)
            groups.add(key[:2])
    expected_keys = adapter.contract["e1"]["test_appearances_per_arm"] * len(MODELS)
    if groups != expected_groups or len(keys) != expected_keys:
        raise BindingError("canonical literal target coverage does not match the frozen baseline")
    return {"status": "ALL_LITERAL_TARGET_TEXT_EQUAL", "same_keys_checked": len(keys), "baseline_model_groups": len(groups), "literal_target_text_differences": 0, "comparison": "canonical CSV y_true versus feature y_true_serialized, string equality before any fit", "scores_computed": False, "new_fits": 0}


def check(adapter: Adapter) -> dict:
    adapter.load_and_verify_freeze(adapter.original / "analysis_contract.json", adapter.original / "stage_a_freeze.json")
    bound = {}
    for sid in adapter.registry["sources"]:
        path = adapter.source_path(sid)
        adapter.verify_source(adapter.contract, sid, path)
        bound[sid] = {"path": str(path), **adapter.file_identity(path)}
    return {"status": "PORTABLE_INPUT_BINDINGS_PASS", "sources": bound, "literal_target_gate": literal_target_check(adapter), "scientific_execution_performed": False}


def inspect_saved(adapter: Adapter) -> dict:
    with (adapter.original / "results/e1_run_01/run_status.json").open(encoding="utf-8") as handle:
        status = json.load(handle)
    if status["status"] != "COMPLETED":
        raise BindingError("preserved E1 completion record is not complete")
    sampled_csv = {}
    for relative in ("results/e1_run_01/block_00__baseline/predictions.csv", "results/e1_model_metrics.csv", "results/e1_block_arm_results.csv", "results/e2_run_01/decomposition.csv", "results/e2_run_01/host_support.csv", "results/e2_run_01/paired_panel.csv"):
        with (adapter.original / relative).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            sample = list(itertools.islice(reader, 2))
            if not sample:
                raise BindingError("saved output sample is empty: " + relative)
            sampled_csv[relative] = {"columns": reader.fieldnames, "records_accessed": len(sample), "sample": sample}
    with (adapter.original / "results/e2_run_01/summary.json").open(encoding="utf-8") as handle:
        e2 = json.load(handle)
    return {"status": "SAVED_OUTPUT_ACCESS_PASS", "e1_completion_record": {k: status[k] for k in ("status", "completed_jobs", "actual_rf_fits_completed")}, "e2_summary_keys": list(e2), "saved_csv_samples": sampled_csv, "scores_computed": False, "new_fits": 0, "scope": "sample access to preserved outputs; no whole-run rescore"}


def load_original(adapter: Adapter, filename: str):
    helper = types.ModuleType("staged_io")
    for name in ("load_and_verify_freeze", "new_output_dir", "verify_source", "file_identity"):
        setattr(helper, name, getattr(adapter, name))
    helper.write_json = write_json
    sys.modules["staged_io"] = helper
    spec = importlib.util.spec_from_file_location("preserved_" + Path(filename).stem, adapter.original / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_e1(adapter: Adapter, output: Path) -> None:
    # The extra text gate completes before loading/calling the original execute.
    literal_target_check(adapter)
    original = load_original(adapter, "e1_parent_purge_runner.py")
    args = argparse.Namespace(contract=str(adapter.original / "analysis_contract.json"), freeze_receipt=str(adapter.original / "stage_a_freeze.json"), output_dir=str(output))
    for sid, argument in original.SOURCE_ARGUMENTS:
        setattr(args, argument, str(adapter.source_path(sid)))
    original.execute(args)


def run_e2(adapter: Adapter, output: Path) -> int:
    original = load_original(adapter, "e2_rank_diagnostic.py")
    return original.main(["--contract", str(adapter.original / "analysis_contract.json"), "--freeze-receipt", str(adapter.original / "stage_a_freeze.json"), "--membership-csv", str(adapter.source_path("family_membership")), "--predictions-csv", str(adapter.source_path("family_predictions")), "--output-dir", str(output)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "inspect-saved", "rescore-e1", "run-e1", "run-e2"))
    parser.add_argument("--snapshot-root", default=Path(__file__).resolve().parents[2], type=Path, help="repository root containing the fixed snapshot data/ and results/; defaults to this checkout root")
    parser.add_argument("--output-dir", type=Path, help="new absolute directory outside the preserved inputs; execution commands only")
    args = parser.parse_args()
    if (args.command.startswith("run-") or args.command == "rescore-e1") != (args.output_dir is not None):
        parser.error("only rescore-e1/run-e1/run-e2 require --output-dir")
    try:
        adapter = Adapter(Path(__file__).resolve().parents[1], args.snapshot_root)
        if args.command == "run-e1":
            # These settings apply only to this explicitly requested child process.
            for key, value in adapter.contract["execution"]["thread_environment"].items():
                os.environ[key] = value
        checked_preflight = None
        if args.command == "rescore-e1":
            from rescore_e1 import preflight
            checked_preflight = preflight(adapter, load_original)
        bindings = check(adapter)
        if args.command == "check":
            print(json.dumps(bindings, indent=2, allow_nan=False))
        elif args.command == "inspect-saved":
            print(json.dumps(inspect_saved(adapter), indent=2, allow_nan=False))
        elif args.command == "run-e1":
            run_e1(adapter, args.output_dir)
        elif args.command == "rescore-e1":
            from rescore_e1 import rescore
            print(json.dumps(rescore(adapter, args.output_dir, load_original, checked_preflight), indent=2, allow_nan=False))
        else:
            return run_e2(adapter, args.output_dir)
        return 0
    except (BindingError, OSError, ValueError, KeyError, ImportError) as exc:
        print("STOP: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
