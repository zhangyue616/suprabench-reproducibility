"""Bounded E1 CPU runner; real execution requires the complete frozen contract.

Canonical CPU formulas are persisted here rather than importing the old runner.
The only models are PAIR_RF and the six prespecified nulls. Frozen masks are
consumed as membership records; parent identities and folds are never rebuilt.
--self-test reads no real inputs and performs no estimator fit.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARMS = ("baseline", "parent_purge", "random_purge_0", "random_purge_1", "random_purge_2")
NULLS = ("sim_knn_k5", "additive", "host_only_RF", "guest_only_RF", "cond_only_RF", "source_only_RF")
MODELS = ("PAIR_RF", *NULLS)
RF_FEATURES = (
    ("PAIR_RF", "X_pair"), ("host_only_RF", "X_host"),
    ("guest_only_RF", "X_guest"), ("cond_only_RF", "X_condition"),
    ("source_only_RF", "X_source"),
)
RF_RECIPE = {"n_estimators": 300, "n_jobs": 1, "random_state": 0, "max_features": "sqrt"}
THREAD_VARIABLES = ("BLIS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
PREDICTION_SCHEMA = ("split", "block_id", "fold_index", "seed", "model", "row_index", "host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256", "y_true", "y_pred")
METRIC_SCHEMA = ("scope", "split", "block_id", "fold_index", "seed", "model", "n", "mae", "r2", "spearman", "r2_status", "spearman_status")
BLOCK_MAP_SCHEMA = ("b", "block_id", "seed", "fold_index")
MASK_SCHEMA = (
    "b", "block_id", "seed", "fold_index", "arm", "row_index", "original_role",
    "retained_train", "deleted_train", "host_id", "guest_id",
    "canonical_host_sha256", "canonical_guest_sha256", "parent_host_key", "parent_guest_key",
    "host_uncertain", "guest_uncertain", "host_parent_seen_in_test", "guest_parent_seen_in_test", "mask_seed",
)
FEATURE_KEYS = ("row_index", "y", "y_true_serialized", "row_host_id", "row_guest_id", "host_index", "guest_index", "X_pair", "X_host", "X_guest", "X_condition", "X_source", "HostSim", "GuestSim")
SOURCE_ARGUMENTS = (
    ("canonical_features", "features_npz"),
    ("canonical_baseline_predictions", "baseline_predictions"),
    ("canonical_baseline_metrics", "baseline_metrics"),
    ("e1_train_masks", "masks"), ("e1_block_map", "block_map"),
)


class GateError(RuntimeError):
    """An E1 fail-closed boundary or implementation-check failure."""


@dataclass
class Job:
    b: int
    block_id: str
    fold_index: int
    seed: str
    arm: str
    train: list[int]
    test: list[int]
    neither: list[int]
    deleted: list[int]
    test_records: list[dict[str, str]]


@dataclass
class FitCounter:
    attempted: int = 0
    completed: int = 0

    def begin(self) -> None:
        if self.attempted >= 300:
            raise GateError("RF fit budget exhausted; no additional fit is allowed")
        self.attempted += 1


def _read_records(path: Path, schema: tuple[str, ...], delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if tuple(reader.fieldnames or ()) != schema:
            raise GateError(f"input schema mismatch: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise GateError(f"malformed input record: {path}")
    return rows


def _integer(text: str, label: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise GateError(f"invalid integer: {label}") from exc
    if str(value) != text:
        raise GateError(f"noncanonical integer serialization: {label}")
    return value


def _boolean(text: str, label: str) -> bool:
    if text not in ("0", "1"):
        raise GateError(f"invalid boolean serialization: {label}")
    return text == "1"


def _load_jobs(masks_path: Path, block_map_path: Path, n_rows: int) -> list[Job]:
    """Validate frozen membership mechanically, without recomputing any rule."""
    block_records = _read_records(block_map_path, BLOCK_MAP_SCHEMA)
    if len(block_records) != 12:
        raise GateError("E1 requires exactly the twelve canonical double-cold blocks")
    block_by_b: dict[int, dict[str, str]] = {}
    folds = set()
    for row in block_records:
        b = _integer(row["b"], "block b")
        fold = _integer(row["fold_index"], "block fold_index")
        if b in block_by_b or fold in folds:
            raise GateError("duplicate frozen block mapping")
        if row["seed"] != str(fold) or row["block_id"] != f"double_cold/seed_{fold}":
            raise GateError("frozen block identity differs from canonical CPU convention")
        block_by_b[b] = row
        folds.add(fold)
    if folds != set(range(12)):
        raise GateError("canonical double-cold fold indices must be 0 through 11")
    grouped: dict[tuple[int, str], dict[int, dict[str, str]]] = {}
    for row in _read_records(masks_path, MASK_SCHEMA):
        b = _integer(row["b"], "mask b")
        if b not in block_by_b or row["arm"] not in ARMS:
            raise GateError("mask record is outside the frozen block/arm roster")
        if any(row[key] != block_by_b[b][key] for key in BLOCK_MAP_SCHEMA):
            raise GateError("mask and frozen block map disagree")
        index = _integer(row["row_index"], "mask row_index")
        if index < 0 or index >= n_rows:
            raise GateError("mask row index is outside the frozen feature row authority")
        for key in ("host_id", "guest_id"):
            _integer(row[key], key)
        for key in ("retained_train", "deleted_train", "host_uncertain", "guest_uncertain", "host_parent_seen_in_test", "guest_parent_seen_in_test"):
            _boolean(row[key], key)
        original = row["original_role"]
        retained = _boolean(row["retained_train"], "retained_train")
        deleted = _boolean(row["deleted_train"], "deleted_train")
        if original not in ("train", "test", "neither"):
            raise GateError("unknown original canonical role")
        if original == "train":
            if retained == deleted:
                raise GateError("original train row must be retained or deleted exactly once")
        elif retained or deleted:
            raise GateError("test/neither row may not enter or be deleted from training")
        if row["arm"] == "baseline" and (retained != (original == "train") or deleted):
            raise GateError("baseline mask altered canonical training membership")
        rows = grouped.setdefault((b, row["arm"]), {})
        if index in rows:
            raise GateError("duplicate frozen mask row")
        rows[index] = row
    expected_groups = {(b, arm) for b in block_by_b for arm in ARMS}
    if set(grouped) != expected_groups:
        raise GateError("frozen masks do not contain the exact 12 by 5 job roster")
    universe = set(range(n_rows))
    jobs = []
    identity_keys = ("original_role", "host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256")
    for b in sorted(block_by_b, key=lambda item: int(block_by_b[item]["fold_index"])):
        baseline = grouped[(b, "baseline")]
        for arm in ARMS:
            records = grouped[(b, arm)]
            if set(records) != universe:
                raise GateError("each block/arm must retain a record for every original feature row")
            for index, row in records.items():
                if any(row[key] != baseline[index][key] for key in identity_keys):
                    raise GateError("frozen arm changed an original row identity or canonical role")
            train = sorted(index for index, row in records.items() if row["retained_train"] == "1")
            test = sorted(index for index, row in records.items() if row["original_role"] == "test")
            neither = sorted(index for index, row in records.items() if row["original_role"] == "neither")
            deleted = sorted(index for index, row in records.items() if row["deleted_train"] == "1")
            if len(train) < 50 or len(test) < 25:
                raise GateError("frozen job violates canonical minimum train/test membership")
            info = block_by_b[b]
            jobs.append(Job(b, info["block_id"], int(info["fold_index"]), info["seed"], arm, train, test, neither, deleted, [records[index] for index in test]))
        parent_count = len(next(job.deleted for job in jobs if job.b == b and job.arm == "parent_purge"))
        if any(len(job.deleted) != parent_count for job in jobs if job.b == b and job.arm.startswith("random_purge_")):
            raise GateError("random purge training loss does not match its frozen parent-purge control")
    for arm in ARMS:
        if sum(len(job.test) for job in jobs if job.arm == arm) != 1716:
            raise GateError("the complete test set must preserve 1716 evaluation appearances per arm")
    return jobs


def _load_features(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as loaded:
        if not set(FEATURE_KEYS).issubset(loaded.files):
            raise GateError("frozen feature bundle is missing an E1 input array")
        features = {key: loaded[key] for key in FEATURE_KEYS}
    n_rows = len(features["row_index"])
    if features["row_index"].tolist() != list(range(n_rows)):
        raise GateError("feature physical row authority is not contiguous zero-based row_index")
    for key in ("y", "y_true_serialized", "row_host_id", "row_guest_id", "host_index", "guest_index"):
        if features[key].shape != (n_rows,):
            raise GateError(f"feature row shape mismatch: {key}")
    for key in ("row_index", "row_host_id", "row_guest_id", "host_index", "guest_index"):
        if not np.issubdtype(features[key].dtype, np.integer):
            raise GateError(f"integer authority dtype required: {key}")
    for _, key in RF_FEATURES:
        if features[key].ndim != 2 or features[key].shape[0] != n_rows or not features[key].shape[1]:
            raise GateError(f"feature matrix shape mismatch: {key}")
    for key in ("y", *(name for _, name in RF_FEATURES), "HostSim", "GuestSim"):
        if not np.issubdtype(features[key].dtype, np.number) or not np.isfinite(features[key]).all():
            raise GateError(f"nonfinite or nonnumeric model input: {key}")
    for prefix in ("host", "guest"):
        matrix = features["HostSim" if prefix == "host" else "GuestSim"]
        indices = features[f"{prefix}_index"]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (indices < 0).any() or (indices >= matrix.shape[0]).any():
            raise GateError(f"similarity/index authority mismatch: {prefix}")
    return features


def _rf(features, y, train, test):
    from sklearn.ensemble import RandomForestRegressor

    # Source-faithful list indexing and stored dtypes; no pre-fit cast or copy.
    return RandomForestRegressor(300, n_jobs=1, random_state=0, max_features="sqrt").fit(features[train], y[train]).predict(features[test])


def _additive(host_prediction, guest_prediction, y, train):
    return host_prediction + guest_prediction - y[train].mean()


def _sim_knn(host_index, guest_index, host_similarity, guest_similarity, y, train, test, k=5):
    import numpy as np

    if not train:
        raise GateError("k5 requires nonempty frozen training membership")
    train_host = host_index[train]
    train_guest = guest_index[train]
    train_y = y[train]
    prediction = np.zeros(len(test))
    max_similarity = np.zeros(len(test))
    for output_index, row_index in enumerate(test):
        combined = (host_similarity[host_index[row_index], train_host] + guest_similarity[guest_index[row_index], train_guest]) / 2.0
        selected = np.argpartition(-combined, min(k, len(combined) - 1))[:k]
        weights = combined[selected] + 1e-6
        prediction[output_index] = np.sum(weights * train_y[selected]) / np.sum(weights)
        max_similarity[output_index] = combined[selected].max()
    return prediction, max_similarity


def _metric_values(y_true, y_pred) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    if len(y_true) != len(y_pred) or not len(y_true) or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise GateError("nonfinite, empty, or unequal-length scoring input")
    mae = float(mean_absolute_error(y_true, y_pred))
    target_varied = len(y_true) >= 2 and len(np.unique(y_true)) > 1
    prediction_varied = np.std(y_pred) > 1e-9
    standard = float(r2_score(y_true, y_pred)) if target_varied else None
    guarded = standard if target_varied and prediction_varied else None
    if standard is not None and not math.isfinite(standard):
        standard = guarded = None
    reason = None if guarded is not None else ("TARGET_CONSTANT_OR_TOO_SHORT" if not target_varied else ("CONSTANT_PREDICTION_GUARD" if not prediction_varied else "NONFINITE_R2"))
    if target_varied:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = float(spearmanr(y_true, y_pred).statistic)
        spearman = candidate if math.isfinite(candidate) else None
    else:
        spearman = None
    return {
        "n": len(y_true), "mae": mae, "mae_status": "DEFINED",
        "guarded_r2": guarded, "guarded_r2_status": "DEFINED" if guarded is not None else "UNKNOWN",
        "guarded_r2_undefined_reason": reason,
        "standard_r2": standard, "standard_r2_status": "DEFINED" if standard is not None else "UNKNOWN",
        "standard_r2_undefined_reason": None if standard is not None else ("TARGET_CONSTANT_OR_TOO_SHORT" if not target_varied else "NONFINITE_R2"),
        "spearman": spearman, "spearman_status": "DEFINED" if spearman is not None else "UNKNOWN",
        "spearman_undefined_reason": None if spearman is not None else ("TARGET_CONSTANT_OR_TOO_SHORT" if not target_varied else "NONFINITE_SPEARMAN"),
    }


def _margin(metrics: dict[str, dict[str, Any]], kind: str) -> dict[str, Any]:
    field = f"{kind}_r2"
    eligible = [model for model in NULLS if metrics[model][field] is not None and math.isfinite(metrics[model][field])]
    ineligible = {model: metrics[model][f"{field}_undefined_reason"] for model in NULLS if model not in eligible}
    winner = max(eligible, key=lambda model: metrics[model][field]) if eligible else None
    pair = metrics["PAIR_RF"][field]
    best = metrics[winner][field] if winner is not None else None
    value = pair - best if pair is not None and best is not None else None
    return {"margin": value, "status": "DEFINED" if value is not None else "UNKNOWN", "best_null": winner, "best_null_r2": best, "eligible_nulls": eligible, "ineligible_nulls": ineligible, "undefined_reason": None if value is not None else ("PAIR_RF_UNDEFINED" if pair is None else "NO_ELIGIBLE_NULL")}


def _roundtrip_equal(reference: float, candidate: float) -> bool:
    return math.isfinite(reference) and math.isfinite(candidate) and abs(reference - candidate) <= 8 * math.ulp(reference)


def _baseline_inputs(predictions_path: Path, metrics_path: Path, jobs: list[Job], features: dict[str, Any]):
    predictions = {}
    for row in _read_records(predictions_path, PREDICTION_SCHEMA):
        if row["split"] != "double_cold" or row["model"] not in MODELS:
            continue
        key = (row["block_id"], row["model"], _integer(row["row_index"], "baseline row_index"))
        if key in predictions:
            raise GateError("duplicate canonical baseline prediction key")
        predictions[key] = row
    metric_rows = {}
    for row in _read_records(metrics_path, METRIC_SCHEMA, "\t"):
        if row["scope"] != "model_block" or row["split"] != "double_cold" or row["model"] not in MODELS:
            continue
        key = (row["block_id"], row["model"])
        if key in metric_rows:
            raise GateError("duplicate canonical baseline metric key")
        metric_rows[key] = row
    expected_predictions = {(job.block_id, model, index) for job in jobs if job.arm == "baseline" for model in MODELS for index in job.test}
    expected_metrics = {(job.block_id, model) for job in jobs if job.arm == "baseline" for model in MODELS}
    if set(predictions) != expected_predictions or set(metric_rows) != expected_metrics:
        raise GateError("canonical baseline keys do not match the frozen complete test membership")
    for job in (item for item in jobs if item.arm == "baseline"):
        for row in job.test_records:
            index = int(row["row_index"])
            for model in MODELS:
                original = predictions[(job.block_id, model, index)]
                if original["fold_index"] != str(job.fold_index) or original["seed"] != job.seed:
                    raise GateError("canonical baseline block/seed identity mismatch")
                if any(original[key] != row[key] for key in ("host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256")):
                    raise GateError("canonical baseline prediction row identity mismatch")
                if not _roundtrip_equal(float(original["y_true"]), float(str(features["y_true_serialized"][index]))):
                    raise GateError(f"canonical baseline serialized target identity mismatch at {job.block_id}/{model}/row_{index}")
                if not math.isfinite(float(original["y_pred"])):
                    raise GateError("canonical baseline prediction is nonfinite")
    return predictions, metric_rows


def _compare_baseline(job: Job, predictions: dict[str, Any], metrics: dict[str, Any], features, old_predictions, old_metrics) -> dict[str, Any]:
    compared = {}
    passed = True
    for model in MODELS:
        maximum = 0.0
        mismatched_rows = []
        for index, value in zip(job.test, predictions[model], strict=True):
            old = old_predictions[(job.block_id, model, index)]
            expected = float(old["y_pred"])
            actual = float(value)
            maximum = max(maximum, abs(actual - expected))
            if not _roundtrip_equal(expected, actual):
                mismatched_rows.append({"row_index": index, "expected": old["y_pred"], "actual": format(actual, ".17g"), "absolute_difference": abs(actual - expected)})
        old = old_metrics[(job.block_id, model)]
        new = metrics[model]
        mismatched_metrics = []
        if old["n"] != str(len(job.test)) or old["fold_index"] != str(job.fold_index) or old["seed"] != job.seed:
            mismatched_metrics.append({"field": "membership_identity", "status": "MISMATCH"})
        for original_field, new_field in (("mae", "mae"), ("r2", "guarded_r2"), ("spearman", "spearman")):
            old_value, new_value = old[original_field], new[new_field]
            same = old_value == "" and new_value is None
            if old_value != "" and new_value is not None:
                same = _roundtrip_equal(float(old_value), float(new_value))
            if not same:
                mismatched_metrics.append({"field": original_field, "expected": old_value, "actual": new_value})
        for old_field, new_field in (("r2_status", "guarded_r2_status"), ("spearman_status", "spearman_status")):
            if old[old_field] != new[new_field]:
                mismatched_metrics.append({"field": old_field, "expected": old[old_field], "actual": new[new_field]})
        model_pass = not mismatched_rows and not mismatched_metrics
        passed = passed and model_pass
        compared[model] = {"status": "PASS" if model_pass else "MISMATCH", "max_absolute_prediction_difference": maximum, "prediction_mismatches": mismatched_rows, "metric_mismatches": mismatched_metrics, "same_target": True, "target_comparison_source": "frozen y_true_serialized against canonical prediction CSV", "scoring_target_source": "original stored feature y dtype"}
    return {"block_id": job.block_id, "status": "PASS" if passed else "STOP_BASELINE_IMPLEMENTATION_MISMATCH", "comparison_rule": "finite numeric roundtrip: abs(new-old) <= 8*ulp(old); no tolerance widening", "models": compared}


def _peak_memory() -> dict[str, Any]:
    """Report process high-water memory without adding a dependency."""
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            process_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            process_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
            process_info.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not process_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error())
            return {"status": "MEASURED", "bytes": int(counters.PeakWorkingSetSize), "scope": "process_lifetime_peak_working_set", "method": "Windows GetProcessMemoryInfo"}
        except (OSError, AttributeError, ImportError) as exc:
            return {"status": "UNKNOWN", "bytes": None, "reason": type(exc).__name__}
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"status": "MEASURED", "bytes": value if sys.platform == "darwin" else value * 1024, "scope": "process_lifetime_peak_resident_set", "method": "stdlib resource"}
    except (ImportError, AttributeError, OSError) as exc:
        return {"status": "UNKNOWN", "bytes": None, "reason": type(exc).__name__}


def _write_predictions(path: Path, job: Job, predictions: dict[str, Any], features) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_SCHEMA, lineterminator="\n")
        writer.writeheader()
        for model in MODELS:
            for row, value in zip(job.test_records, predictions[model], strict=True):
                index = int(row["row_index"])
                writer.writerow({"split": "double_cold", "block_id": job.block_id, "fold_index": str(job.fold_index), "seed": job.seed, "model": model, "row_index": str(index), "host_id": row["host_id"], "guest_id": row["guest_id"], "canonical_host_sha256": row["canonical_host_sha256"], "canonical_guest_sha256": row["canonical_guest_sha256"], "y_true": str(features["y_true_serialized"][index]), "y_pred": format(float(value), ".17g")})
        handle.flush()
        os.fsync(handle.fileno())


def _equal_block_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(arm, fold) for arm in ARMS for fold in range(12)}
    keys = [(row["arm"], row["fold_index"]) for row in results]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise GateError("summary requires all sixty distinct completed frozen jobs")
    summary: dict[str, Any] = {"status": "COMPLETE", "n_blocks": 12, "weighting": "twelve blocks equally weighted; random arms are matched controls, not independent blocks", "arms": {}}
    for arm in ARMS:
        rows = sorted((row for row in results if row["arm"] == arm), key=lambda row: row["fold_index"])
        arm_summary = {}
        for kind in ("guarded", "standard"):
            values = [row[f"{kind}_margin"]["margin"] for row in rows]
            unknown = [row["block_id"] for row in rows if row[f"{kind}_margin"]["margin"] is None]
            mean = None if unknown else sum(values) / 12
            arm_summary[f"{kind}_margin_equal_block_mean"] = {"value": mean, "status": "DEFINED" if mean is not None else "UNKNOWN", "undefined_blocks": unknown, "denominator_blocks": 12}
        summary["arms"][arm] = arm_summary
    for kind in ("guarded", "standard"):
        field = f"{kind}_margin_equal_block_mean"
        means = {arm: summary["arms"][arm][field]["value"] for arm in ARMS}
        primary_arms = ("parent_purge", "random_purge_0", "random_purge_1", "random_purge_2")
        primary = None if any(means[arm] is None for arm in primary_arms) else means["parent_purge"] - sum(means[arm] for arm in primary_arms[1:]) / 3
        secondary = None if means["parent_purge"] is None or means["baseline"] is None else means["parent_purge"] - means["baseline"]
        summary[f"{kind}_primary_parent_minus_random"] = {"value": primary, "status": "DEFINED" if primary is not None else "UNKNOWN", "formula": "parent_purge equal-block margin mean minus mean of three random-purge equal-block margin means"}
        summary[f"{kind}_secondary_parent_minus_baseline"] = {"value": secondary, "status": "DEFINED" if secondary is not None else "UNKNOWN"}
    return summary


def execute(args: argparse.Namespace) -> None:
    from staged_io import load_and_verify_freeze, new_output_dir, verify_source, write_json

    contract = load_and_verify_freeze(Path(args.contract), Path(args.freeze_receipt))
    if contract["e1"]["rf_recipe"] != RF_RECIPE or contract["e1"]["minimum_train_rows"] != 50:
        raise GateError("E1 frozen RF recipe/minimum training contract mismatch")
    for source_id, argument in SOURCE_ARGUMENTS:
        verify_source(contract, source_id, Path(getattr(args, argument)))
    if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
        raise GateError("canonical CPU execution requires every locked thread environment variable to equal 1")
    features = _load_features(Path(args.features_npz))
    jobs = _load_jobs(Path(args.masks), Path(args.block_map), len(features["row_index"]))
    for job in jobs:
        for row in job.test_records:
            index = int(row["row_index"])
            if int(row["host_id"]) != int(features["row_host_id"][index]) or int(row["guest_id"]) != int(features["row_guest_id"][index]):
                raise GateError("frozen test identity differs from the feature authority")
        if len(set(features["y"][job.test])) < 2:
            raise GateError("canonical test validity requires at least two distinct finite target values")
    output = Path(new_output_dir(contract, Path(args.output_dir)))
    sources = {source_id: contract["sources"][source_id] for source_id, _ in SOURCE_ARGUMENTS}
    provenance = {"sources": sources, "contract_path": str(Path(args.contract).resolve()), "freeze_receipt_path": str(Path(args.freeze_receipt).resolve()), "feature_dtypes": {key: str(value.dtype) for key, value in features.items()}, "train_indexing": "original sorted Python integer lists; model inputs retain stored array dtypes", "standard_auxiliary": "same six nulls, standard R2 allows constant prediction; no additional mean predictor"}
    write_json(output / "run_started.json", {"status": "STARTED", "planned_jobs": 60, "maximum_rf_fits": 300, "arms": ARMS, "models": MODELS, "rf_recipe": RF_RECIPE, "provenance": provenance, "pid": os.getpid(), "platform": platform.platform(), "started_unix_seconds": time.time()})
    counter = FitCounter()
    results = []
    started = time.monotonic()
    # All original-baseline blocks precede intervention fits. A mismatch cannot
    # trigger additional arms, retries, replacement seeds, or a wider tolerance.
    ordered_jobs = [job for arm in ARMS for job in jobs if job.arm == arm]
    active_job_dir = None
    try:
        import numpy as np

        old_predictions, old_metrics = _baseline_inputs(Path(args.baseline_predictions), Path(args.baseline_metrics), jobs, features)
        write_json(output / "baseline_input_check.json", {"status": "PASS", "prediction_keys": len(old_predictions), "metric_keys": len(old_metrics), "same_serialized_targets": True, "scope": "twelve double-cold blocks and seven prespecified models"})
        for job in ordered_jobs:
            active_job_dir = output / f"block_{job.fold_index:02d}__{job.arm}"
            active_job_dir.mkdir(exist_ok=False)
            job_started = time.monotonic()
            attempted_before, completed_before = counter.attempted, counter.completed
            membership = {"b": job.b, "block_id": job.block_id, "seed": job.seed, "fold_index": job.fold_index, "arm": job.arm, "train": job.train, "test": job.test, "neither": job.neither, "deleted_train": job.deleted, "source_mask_id": sources["e1_train_masks"]["sha256"], "source_block_map_id": sources["e1_block_map"]["sha256"]}
            write_json(active_job_dir / "membership.json", membership)
            write_json(active_job_dir / "parameters.json", {"rf_recipe": RF_RECIPE, "models": MODELS, "additional_fit_for_additive": False, "sim_knn_k": 5, "minimum_train_rows": 50, "provenance": provenance})
            predictions = {}
            for model, feature_key in RF_FEATURES:
                counter.begin()
                predictions[model] = _rf(features[feature_key], features["y"], job.train, job.test)
                counter.completed += 1
                write_json(active_job_dir / f"fit_completed_{model}.json", {"status": "FIT_COMPLETED", "model": model, "job_rf_fits_completed": counter.completed - completed_before, "total_rf_fits_completed": counter.completed, "total_rf_fits_attempted": counter.attempted})
            predictions["additive"] = _additive(predictions["host_only_RF"], predictions["guest_only_RF"], features["y"], job.train)
            predictions["sim_knn_k5"], max_similarity = _sim_knn(features["host_index"], features["guest_index"], features["HostSim"], features["GuestSim"], features["y"], job.train, job.test, 5)
            for model in MODELS:
                if len(predictions[model]) != len(job.test) or not np.isfinite(predictions[model]).all():
                    raise GateError(f"nonfinite or wrong-length prediction: {job.block_id}/{job.arm}/{model}")
            metrics = {model: _metric_values(features["y"][job.test], predictions[model]) for model in MODELS}
            result = {"b": job.b, "block_id": job.block_id, "fold_index": job.fold_index, "seed": job.seed, "arm": job.arm, "metrics": metrics, "guarded_margin": _margin(metrics, "guarded"), "standard_margin": _margin(metrics, "standard"), "n_train": len(job.train), "n_test": len(job.test), "n_deleted_train": len(job.deleted), "duration_seconds": time.monotonic() - job_started, "peak_memory": _peak_memory(), "actual_rf_fits_attempted": counter.attempted - attempted_before, "actual_rf_fits_completed": counter.completed - completed_before}
            _write_predictions(active_job_dir / "predictions.csv", job, predictions, features)
            write_json(active_job_dir / "max_pair_similarity.json", {"row_index": job.test, "max_pair_similarity": max_similarity.tolist()})
            write_json(active_job_dir / "metrics.json", result)
            if job.arm == "baseline":
                comparison = _compare_baseline(job, predictions, metrics, features, old_predictions, old_metrics)
                write_json(active_job_dir / "baseline_comparison.json", comparison)
                if comparison["status"] != "PASS":
                    raise GateError("unexplained canonical baseline mismatch; affected E1 stopped without tolerance widening")
            write_json(active_job_dir / "status.json", {"status": "COMPLETED", **{key: result[key] for key in ("duration_seconds", "peak_memory", "actual_rf_fits_attempted", "actual_rf_fits_completed")}})
            results.append(result)
            active_job_dir = None
        if counter.attempted != 300 or counter.completed != 300:
            raise GateError("completed E1 run must contain exactly 300 RF fits")
        summary = _equal_block_summary(results)
        write_json(output / "summary.json", summary)
        write_json(output / "run_status.json", {"status": "COMPLETED", "completed_jobs": len(results), "actual_rf_fits_attempted": counter.attempted, "actual_rf_fits_completed": counter.completed, "duration_seconds": time.monotonic() - started, "peak_memory": _peak_memory()})
    except Exception as exc:
        failure = {"status": "EARLY_STOP", "error_type": type(exc).__name__, "error": str(exc), "completed_jobs": len(results), "actual_rf_fits_attempted": counter.attempted, "actual_rf_fits_completed": counter.completed, "duration_seconds": time.monotonic() - started, "peak_memory": _peak_memory(), "resume_policy": "No automatic resume or refit. Existing output directory is rejected; retain this evidence."}
        if active_job_dir is not None and not (active_job_dir / "status.json").exists():
            write_json(active_job_dir / "status.json", failure)
        if not (output / "run_status.json").exists():
            write_json(output / "run_status.json", failure)
        raise


def synthetic_self_test() -> dict[str, Any]:
    """Synthetic arithmetic and scoring only; deliberately does not call _rf."""
    import numpy as np

    checks = {}
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    perfect = _metric_values(y, np.array([0.0, 1.0, 2.0]))
    assert perfect["guarded_r2"] == 1.0 and perfect["mae"] == 0.0 and perfect["spearman"] == 1.0
    checks["perfect"] = perfect
    constant = _metric_values(y, np.array([1.0, 1.0, 1.0]))
    assert constant["guarded_r2"] is None and constant["standard_r2"] == 0.0 and constant["guarded_r2_undefined_reason"] == "CONSTANT_PREDICTION_GUARD"
    checks["constant_prediction"] = constant
    target_constant = _metric_values(np.array([1.0, 1.0]), np.array([0.0, 2.0]))
    assert target_constant["guarded_r2"] is None and target_constant["standard_r2"] is None
    checks["constant_target"] = target_constant
    short = _metric_values(np.array([1.0]), np.array([1.0]))
    assert short["standard_r2"] is None
    checks["short_target"] = short
    rejected = False
    try:
        _metric_values(y, np.array([0.0, math.nan, 2.0]))
    except GateError:
        rejected = True
    assert rejected
    checks["nonfinite_rejected"] = rejected
    additive = _additive(np.array([2.0, 3.0]), np.array([4.0, 5.0]), np.array([1.0, 3.0, 99.0], dtype=np.float32), [0, 1])
    assert additive.tolist() == [4.0, 6.0]
    checks["additive"] = additive.tolist()
    index = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    similarity = np.eye(7, dtype=np.float32)
    similarity[6, :6] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    k_y = np.array([10, 20, 30, 40, 50, 60, 999], dtype=np.float32)
    prediction, maximum = _sim_knn(index, index, similarity, similarity, k_y, list(range(6)), [6])
    weights = similarity[6, 1:6] + 1e-6
    expected = float(np.sum(weights * k_y[1:6]) / np.sum(weights))
    # This manually ordered expected neighbor set can sum in a different order
    # from argpartition. Only this synthetic check uses the source float32 ULP.
    assert abs(float(prediction[0]) - expected) <= 8 * float(np.spacing(np.float32(expected)))
    assert maximum.tolist() == [float(similarity[6, 5])]
    checks["k5"] = {"actual": prediction.tolist(), "expected": [expected], "max_similarity": maximum.tolist()}
    tiny_prediction, _ = _sim_knn(index, index, similarity, similarity, k_y, [0, 1], [6])
    tiny_weights = similarity[6, [0, 1]] + 1e-6
    tiny_expected = float(np.sum(tiny_weights * k_y[[0, 1]]) / np.sum(tiny_weights))
    assert tiny_prediction.tolist() == [tiny_expected]
    checks["k5_small_synthetic_train"] = {"actual": tiny_prediction.tolist(), "expected": [tiny_expected]}
    synthetic_metrics = {model: dict(constant) for model in MODELS}
    synthetic_metrics["PAIR_RF"] = dict(perfect)
    assert _margin(synthetic_metrics, "guarded")["margin"] is None
    assert _margin(synthetic_metrics, "standard")["margin"] == 1.0
    checks["same_six_constant_null_auxiliary"] = {"guarded": _margin(synthetic_metrics, "guarded"), "standard": _margin(synthetic_metrics, "standard")}
    synthetic_metrics["sim_knn_k5"] = dict(perfect)
    assert _margin(synthetic_metrics, "guarded")["best_null"] == "sim_knn_k5"
    assert _margin(synthetic_metrics, "guarded")["margin"] == 0.0
    rows = []
    means = {"baseline": 1.0, "parent_purge": 2.0, "random_purge_0": 0.0, "random_purge_1": 1.0, "random_purge_2": 2.0}
    for arm in ARMS:
        for fold in range(12):
            rows.append({"arm": arm, "fold_index": fold, "block_id": f"double_cold/seed_{fold}", "guarded_margin": {"margin": means[arm]}, "standard_margin": {"margin": means[arm]}})
    summary = _equal_block_summary(rows)
    assert summary["guarded_primary_parent_minus_random"]["value"] == 1.0
    assert summary["guarded_secondary_parent_minus_baseline"]["value"] == 1.0
    checks["equal_block_arithmetic"] = {"primary_actual": summary["guarded_primary_parent_minus_random"]["value"], "primary_expected": 1.0, "secondary_actual": summary["guarded_secondary_parent_minus_baseline"]["value"], "secondary_expected": 1.0}
    next(row for row in rows if row["arm"] == "random_purge_0")["guarded_margin"]["margin"] = None
    undefined_summary = _equal_block_summary(rows)
    assert undefined_summary["guarded_primary_parent_minus_random"]["value"] is None
    checks["undefined_not_dropped"] = undefined_summary["guarded_primary_parent_minus_random"]
    incomplete_rejected = False
    try:
        _equal_block_summary(rows[:-1])
    except GateError:
        incomplete_rejected = True
    assert incomplete_rejected
    checks["incomplete_summary_rejected"] = incomplete_rejected
    assert _roundtrip_equal(1.0, 1.0 + 8 * math.ulp(1.0))
    assert not _roundtrip_equal(1.0, 1.0 + 9 * math.ulp(1.0))
    checks["eight_ulp_boundary"] = True
    return {"status": "PASS", "checks": checks, "real_inputs_read": False, "rf_fits": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    for argument in ("contract", "freeze-receipt", "features-npz", "masks", "block-map", "baseline-predictions", "baseline-metrics", "output-dir"):
        parser.add_argument(f"--{argument}")
    args = parser.parse_args()
    try:
        if args.self_test:
            if any(value is not None for name, value in vars(args).items() if name != "self_test"):
                parser.error("--self-test accepts no real input paths")
            print(json.dumps(synthetic_self_test(), ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        missing = [name.replace("_", "-") for name, value in vars(args).items() if name != "self_test" and value is None]
        if missing:
            parser.error("real execution requires explicit --" + ", --".join(missing))
        execute(args)
        return 0
    except (GateError, OSError, ValueError, KeyError, AssertionError) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
