"""E2: signed decomposition of pooled Spearman ranks on the full CD panel.

Importing this module performs no file access or analysis.  ``--self-test`` uses
only synthetic arrays.  Normal execution verifies the Stage A freeze and source
identities through staged_io before reading either CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, rankdata, spearmanr


MODELS = ("PAIR_RF", "guest_only_RF")
MEMBERSHIP_COLUMNS = (
    "split", "held_family", "role", "row_index", "host_id", "guest_id", "y"
)
PREDICTION_COLUMNS = (
    "row_index", "split", "fold_or_block", "host_id", "guest_id",
    "y_true", "y_pred", "model"
)
NUMERICAL_CHECK_ATOL = 64.0 * np.finfo(np.float64).eps


class E2ValidationError(ValueError):
    """A contract, source-panel, or mathematical check failed."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise E2ValidationError(message)


def _finite_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _check(array.ndim == 1, f"{name}: expected a one-dimensional array")
    _check(bool(np.isfinite(array).all()), f"{name}: non-finite values are forbidden")
    return array


def _scipy_rho(y: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(y) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        value = float(spearmanr(y, prediction).statistic)
    return value if np.isfinite(value) else None


def _undefined_reason(y: np.ndarray, prediction: np.ndarray) -> str | None:
    if len(y) < 2:
        return "fewer_than_two_records"
    y_constant = np.unique(y).size <= 1
    prediction_constant = np.unique(prediction).size <= 1
    if y_constant and prediction_constant:
        return "constant_target_and_prediction"
    if y_constant:
        return "constant_target"
    if prediction_constant:
        return "constant_prediction"
    return None


def rank_decomposition(y: Any, prediction: Any, hosts: Any) -> dict[str, Any]:
    """Decompose globally ranked covariance; terms retain their signs.

    Host means below are means of the *global* average-tie ranks.  Host-local
    Spearman correlations are a separate secondary analysis.
    """
    target = _finite_array(y, "target")
    predicted = _finite_array(prediction, "prediction")
    host = np.asarray(hosts, dtype=object)
    _check(len(target) > 0, "rank decomposition: empty panel")
    _check(host.ndim == 1 and len(target) == len(predicted) == len(host),
           "rank decomposition: array lengths do not agree")
    _check(all(isinstance(value, str) and value != "" for value in host),
           "rank decomposition: each raw host identifier must be a nonempty string")

    target_rank = rankdata(target, method="average")
    prediction_rank = rankdata(predicted, method="average")
    target_mean = float(target_rank.mean())
    prediction_mean = float(prediction_rank.mean())
    target_centered = target_rank - target_mean
    prediction_centered = prediction_rank - prediction_mean
    denominator = float(np.sqrt(
        np.dot(target_centered, target_centered)
        * np.dot(prediction_centered, prediction_centered)
    ))
    host_target_mean = np.empty(len(target), dtype=np.float64)
    host_prediction_mean = np.empty(len(target), dtype=np.float64)
    between_numerator = 0.0
    within_numerator = 0.0
    for raw_host in pd.unique(host):
        mask = host == raw_host
        target_host_mean = float(target_rank[mask].mean())
        prediction_host_mean = float(prediction_rank[mask].mean())
        host_target_mean[mask] = target_host_mean
        host_prediction_mean[mask] = prediction_host_mean
        between_numerator += int(mask.sum()) * (
            target_host_mean - target_mean
        ) * (prediction_host_mean - prediction_mean)
        within_numerator += float(np.dot(
            target_rank[mask] - target_host_mean,
            prediction_rank[mask] - prediction_host_mean,
        ))

    scipy_value = _scipy_rho(target, predicted)
    reason = _undefined_reason(target, predicted)
    result: dict[str, Any] = {
        "n_records": len(target),
        "n_hosts": len(pd.unique(host)),
        "defined": denominator != 0.0,
        "undefined_reason": reason,
        "rank_denominator": denominator,
        "between": None,
        "within": None,
        "total": None,
        "scipy_spearman": scipy_value,
        "decomposition_residual": None,
        "scipy_check_abs_error": None,
        "global_target_rank": target_rank,
        "global_prediction_rank": prediction_rank,
        "host_mean_global_target_rank": host_target_mean,
        "host_mean_global_prediction_rank": host_prediction_mean,
    }
    if denominator == 0.0:
        _check(reason is not None and scipy_value is None,
               "zero rank denominator disagrees with the independent undefined check")
        return result

    _check(reason is None and scipy_value is not None,
           "nonzero rank denominator disagrees with the independent Spearman check")
    between = float(between_numerator / denominator)
    within = float(within_numerator / denominator)
    total = float(np.dot(target_centered, prediction_centered) / denominator)
    residual = float(between + within - total)
    scipy_error = float(abs(total - scipy_value))
    _check(abs(residual) <= NUMERICAL_CHECK_ATOL,
           f"between + within != pooled rank correlation: residual={residual!r}")
    _check(scipy_error <= NUMERICAL_CHECK_ATOL,
           f"pooled rank correlation != scipy Spearman: error={scipy_error!r}")
    result.update(between=between, within=within, total=total,
                  decomposition_residual=residual, scipy_check_abs_error=scipy_error)
    return result


def target_match_kind(membership_target: float, prediction_target: float) -> str | None:
    """Permit exact equality or an exact float32-to-float64 source round trip.

    No absolute/relative tolerance is used.  Merely rounding both distinct
    float64 values to the same float32 value does not make them a match.
    """
    a, b = float(membership_target), float(prediction_target)
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    if a == b:
        return "exact_float64"
    with np.errstate(over="ignore", invalid="ignore"):
        a32 = float(np.float32(a))
        b32 = float(np.float32(b))
    if np.isfinite(a32) and a32 == b:
        return "membership_float32_roundtrip"
    if np.isfinite(b32) and b32 == a:
        return "prediction_float32_roundtrip"
    return None


def _read_required_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    # Preserve identifier and target tokens; Python float() supplies the final
    # float64 parsing rather than a CSV parser's reduced-precision conversion.
    try:
        frame = pd.read_csv(path, usecols=list(columns), dtype="string",
                            keep_default_na=False)
    except (ValueError, OSError) as exc:
        raise E2ValidationError(f"cannot read required columns from {path}: {exc}") from exc
    frame["source_record_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _parse_row_indices(frame: pd.DataFrame, source: str) -> None:
    values = frame["row_index"].tolist()
    bad = [value for value in values
           if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None]
    _check(not bad, f"{source}: row_index must be a nonnegative integer token; examples={bad[:5]!r}")
    frame["row_index"] = [int(value) for value in values]


def _parse_numeric(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    for column in columns:
        try:
            values = np.asarray([float(value) for value in frame[column]], dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise E2ValidationError(f"{source}: {column} contains a nonnumeric token") from exc
        _check(bool(np.isfinite(values).all()), f"{source}: {column} contains non-finite values")
        frame[column] = values


def _nonempty_identifiers(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    for column in columns:
        values = frame[column].tolist()
        _check(all(isinstance(value, str) and value != "" for value in values),
               f"{source}: {column} contains an empty identifier")


def validate_e2_contract(contract: dict[str, Any]) -> dict[str, Any]:
    e2 = contract.get("e2")
    _check(isinstance(e2, dict), "contract.e2 must be an object")
    for key, expected in (("expected_records", 352), ("expected_hosts", 21),
                          ("host_min_distinct_guests", 5)):
        _check(type(e2.get(key)) is int and e2[key] == expected,
               f"contract.e2.{key} must be exactly {expected}")
    _check(e2.get("models") == list(MODELS),
           f"contract.e2.models must be exactly {list(MODELS)!r}")
    sources = contract.get("sources")
    _check(isinstance(sources, dict), "contract.sources must be an object")
    for source_id in ("family_membership", "family_predictions"):
        source = sources.get(source_id)
        _check(isinstance(source, dict) and all(key in source for key in ("path", "sha256", "bytes")),
               f"contract.sources.{source_id} must contain path, sha256, and bytes")
    return e2


def load_paired_panel(membership_path: Path, predictions_path: Path,
                      expected_records: int = 352, expected_hosts: int = 21) -> pd.DataFrame:
    """Read sources only after the caller has verified the freeze and identities."""
    membership = _read_required_csv(membership_path, MEMBERSHIP_COLUMNS)
    selected = membership.loc[
        (membership["split"] == "family_cold")
        & (membership["held_family"] == "cyclodextrin")
        & (membership["role"] == "test")
    ].copy()
    _parse_row_indices(selected, "CD membership")
    _nonempty_identifiers(selected, ("host_id", "guest_id"), "CD membership")
    _parse_numeric(selected, ("y",), "CD membership")
    _check(len(selected) == expected_records,
           f"CD membership: expected {expected_records} records, found {len(selected)}")
    _check(selected["row_index"].is_unique,
           "CD membership: row_index is not unique; duplicate records are forbidden")
    actual_hosts = int(selected["host_id"].nunique())
    _check(actual_hosts == expected_hosts,
           f"CD membership: expected {expected_hosts} raw hosts, found {actual_hosts}")
    panel = selected.sort_values("row_index", kind="stable").reset_index(drop=True)
    panel = panel.rename(columns={"source_record_index": "membership_source_record_index"})

    predictions = _read_required_csv(predictions_path, PREDICTION_COLUMNS)
    predictions = predictions.loc[predictions["model"].isin(MODELS)].copy()
    _parse_row_indices(predictions, "fixed-model predictions")
    # Select the records by membership identity first.  No fold_or_block value
    # is guessed or used to select the cyclodextrin panel.
    predictions = predictions.loc[predictions["row_index"].isin(panel["row_index"])].copy()
    _check(bool((predictions["split"] == "family_cold").all()),
           "membership-matched predictions contain a split other than family_cold")
    _nonempty_identifiers(predictions, ("host_id", "guest_id", "fold_or_block"),
                          "membership-matched predictions")
    _parse_numeric(predictions, ("y_true", "y_pred"), "membership-matched predictions")
    expected_indices = set(panel["row_index"])
    for model in MODELS:
        model_frame = predictions.loc[predictions["model"] == model].copy()
        _check(len(model_frame) == expected_records and model_frame["row_index"].is_unique,
               f"{model}: expected exactly one prediction per {expected_records} membership records; "
               f"found {len(model_frame)} rows / {model_frame['row_index'].nunique()} unique row_index")
        _check(set(model_frame["row_index"]) == expected_indices,
               f"{model}: prediction row_index set differs from CD membership")
        model_frame = model_frame.set_index("row_index").loc[panel["row_index"]].reset_index()
        for identifier in ("row_index", "host_id", "guest_id", "split"):
            mismatch = model_frame[identifier].to_numpy() != panel[identifier].to_numpy()
            _check(not bool(mismatch.any()),
                   f"{model}: {identifier} disagrees with membership at row_index "
                   f"{panel.loc[mismatch, 'row_index'].tolist()[:10]!r}")
        match_kinds = [target_match_kind(a, b) for a, b in
                       zip(panel["y"], model_frame["y_true"])]
        bad_positions = [position for position, kind in enumerate(match_kinds) if kind is None]
        if bad_positions:
            examples = [
                {"row_index": int(panel.iloc[position]["row_index"]),
                 "membership_y": float(panel.iloc[position]["y"]),
                 "prediction_y_true": float(model_frame.iloc[position]["y_true"])}
                for position in bad_positions[:10]
            ]
            raise E2ValidationError(
                f"{model}: target mismatch; exact equality or exact source float32 round trip "
                f"is required; count={len(bad_positions)}, examples={examples!r}"
            )
        for source_column, output_suffix in (
            ("y_true", "y_true"), ("y_pred", "y_pred"),
            ("fold_or_block", "fold_or_block"),
            ("source_record_index", "prediction_source_record_index"),
        ):
            panel[f"{model}_{output_suffix}"] = model_frame[source_column].to_numpy()
        panel[f"{model}_target_match_kind"] = match_kinds

    cross_model_match = [target_match_kind(a, b) for a, b in
                         zip(panel["PAIR_RF_y_true"], panel["guest_only_RF_y_true"])]
    _check(all(kind is not None for kind in cross_model_match),
           "the two models' matched y_true values disagree beyond exact float32 source round trips")
    return panel


def _tie_support(values: np.ndarray) -> dict[str, Any]:
    _, counts = np.unique(values, return_counts=True)
    return {
        "n_unique": int(len(counts)),
        "has_ties": bool((counts > 1).any()),
        "n_records_in_ties": int(counts[counts > 1].sum()),
        "n_excess_ties": int(len(values) - len(counts)),
        "constant": bool(len(counts) <= 1),
    }


def build_host_support(panel: pd.DataFrame, min_distinct_guests: int = 5) -> pd.DataFrame:
    """Retain every raw host, including all ineligible or undefined hosts."""
    rows: list[dict[str, Any]] = []
    for host_id, group in panel.groupby("host_id", sort=True, dropna=False):
        target = _finite_array(group["y"], "host target")
        distinct_guests = int(group["guest_id"].nunique())
        row: dict[str, Any] = {
            "host_id": str(host_id),
            "n_records": len(group),
            "n_distinct_raw_guests": distinct_guests,
            "row_indices": "|".join(str(int(value)) for value in group["row_index"]),
        }
        row.update({f"y_{key}": value for key, value in _tie_support(target).items()})
        reasons: list[str] = []
        if distinct_guests < min_distinct_guests:
            reasons.append(f"fewer_than_{min_distinct_guests}_distinct_raw_guests")
        rhos: dict[str, float | None] = {}
        for model in MODELS:
            predicted = _finite_array(group[f"{model}_y_pred"], f"host {model} prediction")
            row.update({f"{model}_prediction_{key}": value
                        for key, value in _tie_support(predicted).items()})
            reason = _undefined_reason(target, predicted)
            rho = _scipy_rho(target, predicted)
            _check((reason is None) == (rho is not None),
                   f"host {host_id!r} / {model}: undefined status disagrees with scipy")
            rhos[model] = rho
            row[f"{model}_local_rho"] = rho
            row[f"{model}_rho_defined"] = rho is not None
            row[f"{model}_undefined_reason"] = reason
            if reason is not None:
                reasons.append(f"{model}:{reason}")
        both_defined = all(rhos[model] is not None for model in MODELS)
        row["paired_local_rho_difference"] = (
            float(rhos["PAIR_RF"] - rhos["guest_only_RF"]) if both_defined else None
        )
        row["eligible"] = distinct_guests >= min_distinct_guests and both_defined
        row["ineligibility_reasons"] = "|".join(reasons) if reasons else None
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_panel(panel: pd.DataFrame, min_distinct_guests: int = 5
                  ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate the full-panel primary and complete-host secondary diagnostics."""
    panel_output = panel.copy()
    public_keys = (
        "n_records", "n_hosts", "defined", "undefined_reason", "rank_denominator",
        "between", "within", "total", "scipy_spearman", "decomposition_residual",
        "scipy_check_abs_error",
    )
    decompositions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        result = rank_decomposition(panel["y"], panel[f"{model}_y_pred"], panel["host_id"])
        decompositions[model] = {key: result[key] for key in public_keys}
        rows.append({"model": model, **decompositions[model]})
        panel_output["global_y_rank"] = result["global_target_rank"]
        for result_key, suffix in (
            ("global_prediction_rank", "global_prediction_rank"),
            ("host_mean_global_target_rank", "host_mean_global_y_rank"),
            ("host_mean_global_prediction_rank", "host_mean_global_prediction_rank"),
        ):
            panel_output[f"{model}_{suffix}"] = result[result_key]
        if result["defined"]:
            panel_output[f"{model}_within_contribution"] = (
                (result["global_target_rank"] - result["host_mean_global_target_rank"])
                * (result["global_prediction_rank"] - result["host_mean_global_prediction_rank"])
                / result["rank_denominator"]
            )
        else:
            panel_output[f"{model}_within_contribution"] = None

    support = build_host_support(panel, min_distinct_guests)
    eligible = support.loc[support["eligible"]]
    eligible_means = {
        model: float(eligible[f"{model}_local_rho"].mean()) if len(eligible) else None
        for model in MODELS
    }
    paired_mean = float(eligible["paired_local_rho_difference"].mean()) if len(eligible) else None
    within_defined = all(decompositions[model]["defined"] for model in MODELS)
    within_difference = (
        float(decompositions["PAIR_RF"]["within"] - decompositions["guest_only_RF"]["within"])
        if within_defined else None
    )
    undefined_models = [
        f"{model}:{decompositions[model]['undefined_reason']}" for model in MODELS
        if not decompositions[model]["defined"]
    ]
    summary = {
        "status": "E2_COMPLETED",
        "primary_panel": "all membership-selected family_cold/cyclodextrin/test records",
        "target_source": "family_membership.y",
        "rank_method": "global average-tie ranks for each full-panel target/prediction vector",
        "n_records": len(panel),
        "n_raw_hosts": int(panel["host_id"].nunique()),
        "models": list(MODELS),
        "signed_decomposition": decompositions,
        "PAIR_RF_minus_guest_only_RF_within": within_difference,
        "within_difference_undefined_reason": "|".join(undefined_models) if undefined_models else None,
        "secondary_host_support": {
            "n_all_hosts": len(support),
            "n_eligible_hosts": len(eligible),
            "n_ineligible_hosts": len(support) - len(eligible),
            "eligibility": f"at least {min_distinct_guests} distinct raw guests and both local rho values defined",
            "weighting": "equal weight per eligible raw host",
            "eligible_equal_host_mean_rho": eligible_means,
            "eligible_equal_host_mean_paired_difference": paired_mean,
            "mean_undefined_reason": "no_eligible_hosts" if not len(eligible) else None,
        },
        "numerical_check_absolute_tolerance": NUMERICAL_CHECK_ATOL,
    }
    return summary, pd.DataFrame(rows), support, panel_output


def run_self_test() -> dict[str, Any]:
    """Synthetic boundaries only: no path access, real-data reads, or writes."""
    cases: list[dict[str, Any]] = []

    def rank_case(name: str, y: list[float], predicted: list[float], hosts: list[str],
                  expected: dict[str, Any]) -> None:
        result = rank_decomposition(y, predicted, hosts)
        actual = {key: result[key] for key in expected}
        for key, value in expected.items():
            if isinstance(value, float):
                _check(actual[key] is not None and abs(actual[key] - value) <= NUMERICAL_CHECK_ATOL,
                       f"self-test {name}/{key}: actual={actual[key]!r}, expected={value!r}")
            else:
                _check(actual[key] == value,
                       f"self-test {name}/{key}: actual={actual[key]!r}, expected={value!r}")
        cases.append({"name": name, "actual": actual, "expected": expected})

    rank_case("pure_between_constant_host_levels", [1, 1, 2, 2], [3, 3, 4, 4],
              ["A", "A", "B", "B"], {"between": 1.0, "within": 0.0, "total": 1.0})
    rank_case("pure_within_identical_host_ranks", [1, 2, 1, 2], [1, 2, 1, 2],
              ["A", "A", "B", "B"], {"between": 0.0, "within": 1.0, "total": 1.0})
    rank_case("average_tie_ranks", [1, 1, 2, 3], [1, 2, 2, 3],
              ["A", "A", "B", "B"],
              {"between": 2.0 / 3.0, "within": 1.0 / 6.0, "total": 5.0 / 6.0})
    rank_case("overall_constant_target", [1, 1, 1, 1], [1, 2, 3, 4],
              ["A", "A", "B", "B"],
              {"defined": False, "undefined_reason": "constant_target", "between": None,
               "within": None, "total": None, "scipy_spearman": None})
    rank_case("overall_constant_prediction", [1, 2, 3, 4], [1, 1, 1, 1],
              ["A", "A", "B", "B"],
              {"defined": False, "undefined_reason": "constant_prediction", "total": None})

    synthetic_rows: list[dict[str, Any]] = []
    for host_id, guests, target, pair_prediction in (
        ("H4", ["g1", "g2", "g3", "g4", "g4"], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ("H5", ["g1", "g2", "g3", "g4", "g5"], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ("HC5", ["g1", "g2", "g3", "g4", "g5"], [1, 2, 3, 4, 5], [2, 2, 2, 2, 2]),
        ("HYC5", ["g1", "g2", "g3", "g4", "g5"], [2, 2, 2, 2, 2], [1, 2, 3, 4, 5]),
    ):
        for index in range(5):
            synthetic_rows.append({"row_index": len(synthetic_rows), "host_id": host_id,
                                   "guest_id": guests[index], "y": target[index],
                                   "PAIR_RF_y_pred": pair_prediction[index],
                                   "guest_only_RF_y_pred": 5 - index})
    synthetic = pd.DataFrame(synthetic_rows)
    summary, _, support, _ = analyze_panel(synthetic)
    by_host = support.set_index("host_id")
    actual_support = {
        "all_hosts_retained": len(support),
        "H4_distinct_guests": int(by_host.loc["H4", "n_distinct_raw_guests"]),
        "H4_eligible": bool(by_host.loc["H4", "eligible"]),
        "H5_eligible": bool(by_host.loc["H5", "eligible"]),
        "HC5_eligible": bool(by_host.loc["HC5", "eligible"]),
        "HC5_pair_rho_undefined": bool(pd.isna(by_host.loc["HC5", "PAIR_RF_local_rho"])),
        "HYC5_both_rho_undefined": all(bool(pd.isna(by_host.loc["HYC5", f"{model}_local_rho"]))
                                      for model in MODELS),
        "eligible_host_count": summary["secondary_host_support"]["n_eligible_hosts"],
        "eligible_mean_PAIR_RF_rho": summary["secondary_host_support"]["eligible_equal_host_mean_rho"]["PAIR_RF"],
        "eligible_mean_guest_only_RF_rho": summary["secondary_host_support"]["eligible_equal_host_mean_rho"]["guest_only_RF"],
        "eligible_mean_paired_difference": summary["secondary_host_support"]["eligible_equal_host_mean_paired_difference"],
    }
    expected_support = {"all_hosts_retained": 4, "H4_distinct_guests": 4,
                        "H4_eligible": False, "H5_eligible": True, "HC5_eligible": False,
                        "HC5_pair_rho_undefined": True, "HYC5_both_rho_undefined": True,
                        "eligible_host_count": 1, "eligible_mean_PAIR_RF_rho": 1.0,
                        "eligible_mean_guest_only_RF_rho": -1.0,
                        "eligible_mean_paired_difference": 2.0}
    for key, expected in expected_support.items():
        actual = actual_support[key]
        if isinstance(expected, float):
            _check(actual is not None and abs(actual - expected) <= NUMERICAL_CHECK_ATOL,
                   f"self-test host support/{key}: actual={actual!r}, expected={expected!r}")
        else:
            _check(actual == expected,
                   f"self-test host support/{key}: actual={actual!r}, expected={expected!r}")
    _check("fewer_than_5_distinct_raw_guests" in by_host.loc["H4", "ineligibility_reasons"],
           "self-test: the four-guest host lost its ineligibility reason")
    _check("PAIR_RF:constant_prediction" in by_host.loc["HC5", "ineligibility_reasons"],
           "self-test: the constant-prediction host lost its ineligibility reason")
    cases.append({"name": "five_distinct_guest_boundary_and_constant_hosts",
                  "actual": actual_support, "expected": expected_support})

    within_panel = pd.DataFrame({"row_index": range(4), "host_id": ["A", "A", "B", "B"],
                                 "guest_id": ["g1", "g2", "g1", "g2"], "y": [1, 2, 1, 2],
                                 "PAIR_RF_y_pred": [1, 2, 1, 2],
                                 "guest_only_RF_y_pred": [2, 1, 2, 1]})
    within_summary, _, _, _ = analyze_panel(within_panel)
    actual_difference = within_summary["PAIR_RF_minus_guest_only_RF_within"]
    _check(abs(actual_difference - 2.0) <= NUMERICAL_CHECK_ATOL,
           "self-test: the signed paired within difference must be 2")
    _check(within_summary["secondary_host_support"]["eligible_equal_host_mean_paired_difference"] is None,
           "self-test: an empty eligible-host mean must remain null")
    cases.append({"name": "signed_within_difference_and_empty_eligible_set",
                  "actual": {"within_difference": actual_difference,
                             "empty_eligible_mean": None},
                  "expected": {"within_difference": 2.0, "empty_eligible_mean": None}})

    roundtrip = float(np.float32(1.1))
    target_checks = {
        "exact": target_match_kind(1.1, 1.1),
        "source_float32_roundtrip": target_match_kind(1.1, roundtrip),
        "nearby_float64_rejected": target_match_kind(1.1, float(np.nextafter(1.1, 2.0))),
    }
    expected_target_checks = {"exact": "exact_float64",
                              "source_float32_roundtrip": "membership_float32_roundtrip",
                              "nearby_float64_rejected": None}
    _check(target_checks == expected_target_checks,
           f"self-test exact target matching: {target_checks!r}")
    cases.append({"name": "exact_target_identity_without_tolerance",
                  "actual": target_checks, "expected": expected_target_checks})
    return {"status": "SELF_TEST_PASS", "synthetic_cases": len(cases), "cases": cases}


def _source_bindings(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        source_id: {key: contract["sources"][source_id][key] for key in ("path", "sha256", "bytes")}
        for source_id in ("family_membership", "family_predictions")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--membership-csv", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true",
                        help="synthetic boundaries only; no file reads or writes")
    args = parser.parse_args(argv)
    normal_arguments = (args.contract, args.freeze_receipt, args.membership_csv,
                        args.predictions_csv, args.output_dir)
    if args.self_test:
        if any(value is not None for value in normal_arguments):
            parser.error("--self-test must be used alone; it cannot accept real-source paths")
        try:
            print(json.dumps(run_self_test(), indent=2, allow_nan=False))
            return 0
        except (E2ValidationError, ValueError, TypeError) as exc:
            print(json.dumps({"status": "SELF_TEST_FAIL", "error": str(exc)}), file=sys.stderr)
            return 2
    if not all(value is not None for value in normal_arguments):
        parser.error("normal execution requires explicit --contract --freeze-receipt "
                     "--membership-csv --predictions-csv --output-dir")

    try:
        # Shared helper identity and API come from staged_io (Stage A).
        # This import is deliberately bypassed by synthetic self-tests.
        from staged_io import load_and_verify_freeze, new_output_dir, verify_source, write_json

        contract = load_and_verify_freeze(args.contract, args.freeze_receipt)
        e2 = validate_e2_contract(contract)
        verify_source(contract, "family_membership", args.membership_csv)
        verify_source(contract, "family_predictions", args.predictions_csv)
        panel = load_paired_panel(args.membership_csv, args.predictions_csv,
                                  e2["expected_records"], e2["expected_hosts"])
        summary, decomposition, support, paired_panel = analyze_panel(
            panel, e2["host_min_distinct_guests"]
        )
        bindings = _source_bindings(contract)
        summary["inputs"] = bindings
        summary["contract_path"] = str(args.contract.resolve())
        summary["freeze_receipt_path"] = str(args.freeze_receipt.resolve())
        summary["target_match_counts"] = {
            model: {str(key): int(value) for key, value in
                    panel[f"{model}_target_match_kind"].value_counts().items()}
            for model in MODELS
        }
        for frame in (decomposition, support, paired_panel):
            for source_id, identity in bindings.items():
                frame[f"{source_id}_sha256"] = identity["sha256"]
                frame[f"{source_id}_bytes"] = identity["bytes"]
        # Calculation and input validation finish before the first output write.
        output_dir = new_output_dir(contract, args.output_dir)
        for filename, frame in (("decomposition.csv", decomposition),
                                ("host_support.csv", support),
                                ("paired_panel.csv", paired_panel)):
            frame.to_csv(output_dir / filename, index=False, mode="x",
                         float_format="%.17g", na_rep="")
        write_json(output_dir / "summary.json", summary)
        print(json.dumps({"status": "E2_COMPLETED", "n_records": len(panel),
                          "n_raw_hosts": len(support), "output_dir": str(output_dir)},
                         allow_nan=False))
        return 0
    except (E2ValidationError, ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "E2_FAILED", "error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
