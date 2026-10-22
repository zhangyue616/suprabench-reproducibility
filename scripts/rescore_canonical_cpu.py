"""Rescore frozen canonical CPU predictions; never fit or generate predictions."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from reaggregate_canonical_gnn import (
    CPU_MODELS, EXPECTED_POOLED_N, METRIC_COLUMNS, NULL_MODELS,
    PREDICTION_COLUMNS, SPLITS, arrays_from_rows, group_predictions,
    metric_index, production_r2, read_rows, row_keys, selected_rows,
    validate_target_reconstruction,
)
from release_common import (
    ReleaseError, assert_safe_output, atomic_write_json, eight_ulp_equal,
    manifest, repo_root, sha256,
)

ALL_CPU_MODELS = (
    "PAIR_RF", "ECFP_RF", "XGB_pair", "XGB_ecfp", "LGBM_pair", "LGBM_ecfp",
    "3D_pair_RF", *NULL_MODELS,
)


def _number(expected: float, actual: float, label: str) -> dict:
    if not (math.isfinite(expected) and math.isfinite(actual)) or not eight_ulp_equal(expected, actual):
        raise ReleaseError(f"CPU_RESCORE_MISMATCH {label}: expected={expected!r} actual={actual!r}")
    return {"expected": expected, "actual": actual, "absolute_difference": abs(actual - expected),
            "tolerance": 8 * math.ulp(expected)}


def rescore_cpu(root: Path, output: Path, group: dict) -> dict:
    """Called only after the verifier's file and environment checks have passed."""
    assert_safe_output(root, output)
    if output.exists():
        raise ReleaseError(f"CPU_RESCORE_OUTPUT_EXISTS: {output}")
    predictions_path = root / group["run_a"] / "predictions_cpu.csv"
    metrics_path = root / group["run_a"] / "metrics_cpu.tsv"
    rows = read_rows(predictions_path, ",", PREDICTION_COLUMNS)
    stored = read_rows(metrics_path, "\t", METRIC_COLUMNS)
    indexed = metric_index(stored)
    grouped = group_predictions(rows)
    expected_blocks = {
        split: [f"{split}/{'seed' if split == 'double_cold' else 'fold'}_{i}"
                for i in range(12 if split == "double_cold" else 5)]
        for split in SPLITS
    }
    expected_keys = {(split, block, model) for split in SPLITS
                     for block in expected_blocks[split] for model in ALL_CPU_MODELS}
    if set(grouped) != expected_keys:
        raise ReleaseError("CPU_RESCORE_PREDICTION_ROSTER_MISMATCH")
    expected_metrics = {
        ("model_block", split, block, model) for split, block, model in expected_keys
    } | {("model_split_aggregate", split, "ALL", model)
         for split in SPLITS for model in ALL_CPU_MODELS}
    # The stored aggregate uses its original literal block_id; do not infer it.
    aggregate_ids = {row["block_id"] for row in stored if row["scope"] == "model_split_aggregate"}
    if len(aggregate_ids) != 1:
        raise ReleaseError("CPU_RESCORE_AGGREGATE_ID_MISMATCH")
    aggregate_id = next(iter(aggregate_ids))
    expected_metrics = {(scope, split, aggregate_id if scope == "model_split_aggregate" else block, model)
                        for scope, split, block, model in expected_metrics}
    if set(indexed) != expected_metrics or len(rows) != int(group["counts"]["predictions_per_run"]):
        raise ReleaseError("CPU_RESCORE_METRIC_ROSTER_OR_ROW_COUNT_MISMATCH")
    for split, blocks in expected_blocks.items():
        for block in blocks:
            reference = grouped[(split, block, "PAIR_RF")]
            keys = row_keys(reference)
            identity = [{key: row[key] for key in PREDICTION_COLUMNS if key not in ("model", "y_pred")}
                        for row in reference]
            for model in ALL_CPU_MODELS:
                selection = grouped[(split, block, model)]
                if row_keys(selection) != keys or identity != [
                    {key: row[key] for key in PREDICTION_COLUMNS if key not in ("model", "y_pred")}
                    for row in selection
                ]:
                    raise ReleaseError(f"CPU_RESCORE_MEMBERSHIP_MISMATCH: {block}/{model}")
        if sum(len(grouped[(split, block, "PAIR_RF")]) for block in blocks) != EXPECTED_POOLED_N[split]:
            raise ReleaseError(f"CPU_RESCORE_APPEARANCE_COUNT_MISMATCH: {split}")
    target_check = validate_target_reconstruction(
        [rows], root / "data/canonical_cpu/features/tabular_features.npz"
    )
    comparisons, scores = [], {}
    for key, record in indexed.items():
        scope, split, block, model = key
        pooled = scope == "model_split_aggregate"
        selection = selected_rows(rows, split, model) if pooled else grouped[(split, block, model)]
        dtype = np.float32 if model.startswith("XGB_") else np.float64
        y, prediction = arrays_from_rows(selection, dtype, pooled)
        if not np.isfinite(y).all() or not np.isfinite(prediction).all():
            raise ReleaseError(f"CPU_RESCORE_NONFINITE_INPUT: {key}")
        actual, status = production_r2(y, prediction)
        if status != record["r2_status"] or len(selection) != int(record["n"]):
            raise ReleaseError(f"CPU_RESCORE_STATUS_OR_N_MISMATCH: {key}")
        comparison = {"scope": scope, "split": split, "block_id": block, "model": model,
                      "n": len(selection), "status": status}
        if actual is None:
            if record["r2"] != "":
                raise ReleaseError(f"CPU_RESCORE_UNDEFINED_VALUE_MISMATCH: {key}")
        else:
            comparison.update(_number(float(record["r2"]), actual, str(key)))
        comparisons.append(comparison)
        scores[key] = actual

    def margin(split: str, block: str, scope: str) -> dict:
        nulls = {model: scores[(scope, split, block, model)] for model in NULL_MODELS}
        finite = [model for model in NULL_MODELS if nulls[model] is not None]
        if not finite:
            raise ReleaseError(f"CPU_RESCORE_NO_ELIGIBLE_NULL: {split}/{block}")
        best = max(finite, key=lambda model: nulls[model])
        pair = scores[(scope, split, block, "PAIR_RF")]
        if pair is None:
            raise ReleaseError(f"CPU_RESCORE_PAIR_RF_UNDEFINED: {split}/{block}")
        return {"split": split, "block_id": block, "best_null": best,
                "model_r2": pair, "best_null_r2": nulls[best], "delta_r2": pair - nulls[best],
                "null_r2": nulls}

    pooled = [margin(split, aggregate_id, "model_split_aggregate") for split in SPLITS]
    blocks = [margin("double_cold", block, "model_block") for block in expected_blocks["double_cold"]]
    weights = np.asarray([len(grouped[("double_cold", row["block_id"], "PAIR_RF")]) for row in blocks])
    summaries = []
    for aggregation in ("equal_block_mean", "appearance_weighted_mean"):
        reduce = np.mean if aggregation == "equal_block_mean" else lambda values: np.average(values, weights=weights)
        summaries.append({"split": "double_cold", "aggregation": aggregation,
                          **{field: float(reduce([row[field] for row in blocks]))
                             for field in ("model_r2", "best_null_r2", "delta_r2")},
                          "best_null": "PER_BLOCK_MAX_OF_FINITE_SIX",
                          "positive_blocks": sum(row["delta_r2"] > 0 for row in blocks),
                          "total_blocks": len(blocks)})
    with (root / "results/canonical_gnn_aggregation.tsv").open(encoding="utf-8", newline="") as handle:
        saved_summaries = [row for row in csv.DictReader(handle, delimiter="\t") if row["model"] == "PAIR_RF"]
    summary_comparisons = []
    for actual in [dict(row, aggregation="pooled_test_appearances") for row in pooled] + summaries:
        candidates = [row for row in saved_summaries
                      if row["split"] == actual["split"] and row["aggregation"] == actual["aggregation"]]
        if len(candidates) != 1 or candidates[0]["best_null"] != actual["best_null"]:
            raise ReleaseError("CPU_RESCORE_SAVED_SUMMARY_ROSTER_OR_NULL_MISMATCH")
        for field in ("model_r2", "best_null_r2", "delta_r2"):
            summary_comparisons.append({"split": actual["split"], "aggregation": actual["aggregation"],
                                        "field": field, **_number(float(candidates[0][field]), actual[field], field)})
    with (root / "results/canonical_gnn_double_cold_blocks.csv").open(encoding="utf-8", newline="") as handle:
        saved_blocks = list(csv.DictReader(handle))
    if len(saved_blocks) != len(blocks):
        raise ReleaseError("CPU_RESCORE_SAVED_BLOCK_COUNT_MISMATCH")
    for actual, expected in zip(blocks, saved_blocks, strict=True):
        if actual["block_id"] != expected["block_id"] or actual["best_null"] != expected["best_null"]:
            raise ReleaseError("CPU_RESCORE_SAVED_BLOCK_ID_OR_NULL_MISMATCH")
        for actual_field, expected_field in (("model_r2", "PAIR_RF_r2"), ("best_null_r2", "best_null_r2"), ("delta_r2", "PAIR_RF_margin_r2")):
            _number(float(expected[expected_field]), actual[actual_field], expected_field)
    result = {"status": "PASS", "training_performed": False, "prediction_generation_performed": False,
              "scoring": "production guarded R2; float32 targets; block XGBoost predictions float32, other CPU predictions float64; pooled arrays float64",
              "numeric_rule": "abs(actual-expected) <= 8*ulp(expected); exact definedness, keys and memberships",
              "prediction_rows": len(rows), "metric_rows": len(comparisons),
              "defined_r2_rows": sum(row["status"] == "DEFINED" for row in comparisons),
              "target_reconstruction": target_check, "pooled_pair_rf": pooled,
              "double_cold_blocks": blocks, "double_cold_summaries": summaries,
              "metric_comparisons": comparisons, "summary_comparisons": summary_comparisons,
              "code_sha256": sha256(Path(__file__)),
              "scoring_helper_sha256": sha256(Path(__file__).with_name("reaggregate_canonical_gnn.py")),
              "prediction_sha256": sha256(predictions_path), "stored_metrics_sha256": sha256(metrics_path)}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "canonical_cpu_rescore.json", result)
    return {key: result[key] for key in ("status", "training_performed", "prediction_rows", "metric_rows", "defined_r2_rows", "pooled_pair_rf", "double_cold_summaries")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = repo_root(args.root)
    from verify_saved_results import verify_cpu, verify_environment
    spec = manifest(root)
    verify_environment(spec)
    print(verify_cpu(root, spec["scientific_groups"]["canonical_cpu_saved"], args.output.resolve()))


if __name__ == "__main__":
    main()
