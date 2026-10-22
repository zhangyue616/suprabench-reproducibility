"""Rescore all frozen E1 predictions with the preserved scorer, without fitting."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

from staged_io_adapter import BindingError, write_json


def _compare(expected, actual, label: str, counts: dict) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise BindingError("rescore object keys differ: " + label)
        for key in expected:
            _compare(expected[key], actual[key], label + "/" + key, counts)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise BindingError("rescore list shape differs: " + label)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{label}/{index}", counts)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(expected) or not math.isfinite(actual):
            raise BindingError("rescore nonfinite or nonnumeric comparison: " + label)
        difference = abs(actual - expected)
        tolerance = 8 * math.ulp(float(expected))
        if difference > tolerance:
            raise BindingError(f"rescore numeric mismatch: {label}: expected={expected!r} actual={actual!r}")
        counts["numeric_cells"] += 1
        counts["max_absolute_difference"] = max(counts["max_absolute_difference"], difference)
    elif expected != actual:
        raise BindingError(f"rescore exact-value mismatch: {label}: expected={expected!r} actual={actual!r}")
    else:
        counts["exact_cells"] += 1


def preflight(adapter, load_original):
    """Gate code/environment, then self-test before reading real numeric inputs."""
    adapter.load_and_verify_freeze(adapter.original / "analysis_contract.json", adapter.original / "stage_a_freeze.json")
    sys.path.insert(0, str(adapter.snapshot / "scripts"))
    from release_common import assert_safe_output, manifest
    from verify_saved_results import verify_environment
    environment = verify_environment(manifest(adapter.snapshot))
    runner = load_original(adapter, "e1_parent_purge_runner.py")
    self_test = runner.synthetic_self_test()
    if self_test["status"] != "PASS":
        raise BindingError("preserved E1 synthetic scoring self-test failed")
    return environment, runner, self_test


def rescore(adapter, output: Path, load_original, checked_preflight=None) -> dict:
    """Bindings and literal targets are checked by reproduce.py before entry."""
    import numpy as np

    environment, runner, self_test = checked_preflight or preflight(adapter, load_original)
    from release_common import assert_safe_output
    # This small witness retains the source float32 target dtype and literal text;
    # no feature matrix or estimator is needed by the saved-prediction scorer.
    with np.load(adapter.source_path("canonical_features"), allow_pickle=False) as feature:
        features = {key: feature[key] for key in ("row_index", "y", "y_true_serialized")}
    jobs = runner._load_jobs(adapter.source_path("e1_train_masks"), adapter.source_path("e1_block_map"), len(features["row_index"]))
    baseline_predictions, baseline_metrics = runner._baseline_inputs(
        adapter.source_path("canonical_baseline_predictions"),
        adapter.source_path("canonical_baseline_metrics"), jobs, features,
    )
    counts = {"numeric_cells": 0, "exact_cells": 0, "max_absolute_difference": 0.0}
    results, baselines, identities = [], [], {}
    prediction_rows = 0
    for job in jobs:
        directory = adapter.original / "results/e1_run_01" / f"block_{job.fold_index:02d}__{job.arm}"
        predictions_path = directory / "predictions.csv"
        records = runner._read_records(predictions_path, runner.PREDICTION_SCHEMA)
        grouped = {model: [] for model in runner.MODELS}
        for row in records:
            if row["model"] not in grouped:
                raise BindingError("unexpected E1 saved prediction model")
            grouped[row["model"]].append(row)
        predictions, metrics = {}, {}
        for model in runner.MODELS:
            rows = grouped[model]
            if [int(row["row_index"]) for row in rows] != job.test:
                raise BindingError(f"E1 saved prediction test membership/order mismatch: {job.fold_index}/{job.arm}/{model}")
            for row, authority in zip(rows, job.test_records, strict=True):
                index = int(row["row_index"])
                if row["split"] != "double_cold" or row["block_id"] != job.block_id or row["fold_index"] != str(job.fold_index) or row["seed"] != job.seed:
                    raise BindingError("E1 saved prediction block identity mismatch")
                for key in ("host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256"):
                    if row[key] != authority[key]:
                        raise BindingError("E1 saved prediction chemical identity differs from frozen membership")
                if row["y_true"] != str(features["y_true_serialized"][index]):
                    raise BindingError("E1 saved prediction literal target differs from frozen feature target")
            predictions[model] = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
            metrics[model] = runner._metric_values(features["y"][job.test], predictions[model])
        actual = {"metrics": metrics, "guarded_margin": runner._margin(metrics, "guarded"),
                  "standard_margin": runner._margin(metrics, "standard")}
        with (directory / "metrics.json").open(encoding="utf-8") as handle:
            expected = json.load(handle)
        _compare({key: expected[key] for key in actual}, actual, directory.name, counts)
        result = {"arm": job.arm, "fold_index": job.fold_index, "block_id": job.block_id, **actual}
        results.append(result)
        prediction_rows += len(records)
        identities[str(predictions_path.relative_to(adapter.snapshot))] = adapter.file_identity(predictions_path)
        if job.arm == "baseline":
            baseline = runner._compare_baseline(job, predictions, metrics, features, baseline_predictions, baseline_metrics)
            if baseline["status"] != "PASS":
                raise BindingError("E1 saved baseline differs from canonical CPU predictions/metrics within 8 ULP")
            baselines.append(baseline)
    summary = runner._equal_block_summary(results)
    summary_path = adapter.original / "results/e1_run_01/summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        _compare(json.load(handle), summary, "summary", counts)
    with (adapter.original / "results/e1_model_metrics.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    indexed = {(row["block_id"], row["arm"], row["model"]): row for row in table}
    if len(indexed) != 420 or len(table) != 420:
        raise BindingError("E1 saved model-metric table must contain exactly 420 unique rows")
    for job in results:
        for model, metrics in job["metrics"].items():
            row = indexed[(job["block_id"], job["arm"], model)]
            for field, value in metrics.items():
                expected = (float(row[field]) if isinstance(value, (int, float))
                            else None if value is None and row[field] == "" else row[field])
                _compare(expected, value, f"model_metrics/{job['block_id']}/{job['arm']}/{model}/{field}", counts)
    output = output.expanduser()
    if not output.is_absolute():
        raise BindingError("rescore output must be an explicit absolute fresh directory")
    output = output.resolve()
    if output.is_relative_to(adapter.snapshot):
        assert_safe_output(adapter.snapshot, output)
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = adapter.new_output_dir(adapter.contract, output)
    report = {"status": "PASS", "scope": "all 60 saved E1 jobs, 420 model metrics, guarded/standard block margins, equal-block arm means and primary/secondary contrasts",
              "training_performed": False, "new_fits": 0, "prediction_rows": prediction_rows,
              "jobs": len(results), "models_per_job": len(runner.MODELS), "environment": environment,
              "comparison_rule": "finite numeric abs(actual-expected) <= 8*ulp(expected); exact identities, statuses and literal target strings",
              "comparison_counts": counts, "summary": summary,
              "baseline_comparisons": baselines, "synthetic_self_test": self_test,
              "source_identities": identities,
              "scoring_source": adapter.file_identity(adapter.original / "scripts/e1_parent_purge_runner.py"),
              "rescore_source": adapter.file_identity(Path(__file__))}
    write_json(output / "e1_rescore.json", report)
    return {key: report[key] for key in ("status", "scope", "training_performed", "new_fits", "prediction_rows", "jobs", "comparison_counts", "summary")}
