"""Build frozen E1 train masks from the frozen identity-only projections."""

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "inputs" / "identity"
ARMS = ("baseline", "parent_purge", "random_purge_0", "random_purge_1", "random_purge_2")
OUTPUTS = ("train_masks.csv", "block_map.csv", "block_arm_support.csv", "identity_coverage.json")
CONFLICT_FILE = "identity_conflicts.json"
EXPECTED_COLUMNS = {
    "membership": ["split", "fold_index", "seed", "role", "row_index", "host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256"],
    "appearance_keys": ["layer", "host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256", "parent_host_key", "parent_guest_key", "host_uncertain", "guest_uncertain", "uncertain"],
    "aliases": ["entity_type", "raw_id", "canonical_key_sha256"],
    "parent_mapping": ["role", "identity_id", "strict_exact_key", "parent_neutralized_operational_key", "uncertain_reasons"],
    "decisions": ["review_group_id", "decision"],
    "uncertain_links": ["role", "raw_id", "review_group_id"],
}
INPUT_NAMES = {
    "membership": "canonical_membership.csv",
    "appearance_keys": "parent_appearance_keys.csv",
    "aliases": "canonical_aliases.csv",
    "parent_mapping": "parent_mapping.csv",
    "decisions": "parent_decisions.csv",
    "uncertain_links": "uncertain_group_links_v2.csv",
}
MASK_COLUMNS = [
    "b", "block_id", "seed", "fold_index", "arm", "row_index", "original_role",
    "retained_train", "deleted_train", "host_id", "guest_id", "canonical_host_sha256",
    "canonical_guest_sha256", "parent_host_key", "parent_guest_key", "host_uncertain",
    "guest_uncertain", "host_parent_seen_in_test", "guest_parent_seen_in_test", "mask_seed",
]
SUPPORT_COLUMNS = [
    "b", "block_id", "seed", "fold_index", "arm", "original_train_rows", "original_test_rows",
    "original_neither_rows", "retained_train_rows", "deleted_train_rows", "deleted_train_fraction",
    "remaining_unique_raw_hosts", "remaining_unique_raw_guests", "remaining_unique_parent_hosts",
    "remaining_unique_parent_guests", "remaining_host_parent_exposure_rows",
    "remaining_guest_parent_exposure_rows", "remaining_any_parent_exposure_rows",
    "remaining_host_parent_test_intersection_count", "remaining_guest_parent_test_intersection_count",
    "remaining_raw_host_test_intersection_count", "remaining_raw_guest_test_intersection_count",
    "mask_seed", "minimum_train_rows", "min_train_pass", "minimum_test_rows", "min_test_pass",
    "original_train_uncertain_rows", "deleted_train_uncertain_rows", "retained_train_uncertain_rows",
    "deleted_train_host_uncertain_rows", "deleted_train_guest_uncertain_rows",
    "retained_train_host_uncertain_rows", "retained_train_guest_uncertain_rows",
]


class IdentityConflict(Exception):
    def __init__(self, code, details):
        super().__init__(code)
        self.code = code
        self.details = details


def require(condition, code, details):
    if not condition:
        raise IdentityConflict(code, details)


def boolean(value):
    require(value in ("True", "False", "true", "false", "0", "1"), "INVALID_BOOLEAN", {"value": value})
    return value in ("True", "true", "1")


def sha(value, context):
    value = value.lower()
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "INVALID_CANONICAL_SHA", context)
    return value


def read_csv(path, name):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == EXPECTED_COLUMNS[name], "INPUT_SCHEMA_CONFLICT",
                {"input": name, "actual": reader.fieldnames, "expected": EXPECTED_COLUMNS[name]})
        return list(reader)


def put_unique(mapping, key, value, code, context):
    if key in mapping:
        require(mapping[key] == value, code,
                {"key": list(key), "first": mapping[key], "second": value, **context})
    mapping[key] = value


def parent_deleted(train_rows, held_hosts, held_guests):
    return {r["row_index"] for r in train_rows
            if r["parent_host_key"] in held_hosts or r["parent_guest_key"] in held_guests}


def random_deleted(train_row_indices, count, b, j):
    ordered = np.asarray(sorted(train_row_indices), dtype=np.int64)
    seed = 100000 + 100 * b + j
    chosen = np.random.RandomState(seed).choice(ordered, size=count, replace=False)
    return {int(value) for value in chosen}, seed


def self_test():
    rows = [
        {"row_index": 1, "parent_host_key": "H", "parent_guest_key": "x"},
        {"row_index": 2, "parent_host_key": "h", "parent_guest_key": "G"},
        {"row_index": 3, "parent_host_key": "H", "parent_guest_key": "G"},
        {"row_index": 4, "parent_host_key": "h", "parent_guest_key": "g"},
    ]
    actual_or = sorted(parent_deleted(rows, {"H"}, {"G"}))
    require(actual_or == [1, 2, 3], "SELF_TEST_OR_FAIL", {"actual": actual_or})
    first, seed = random_deleted([9, 2, 6, 1, 4], 2, 0, 0)
    second, _ = random_deleted([1, 4, 2, 9, 6], 2, 0, 0)
    require(first == second == {1, 9}, "SELF_TEST_RANDOM_FAIL", {"actual": sorted(first), "second": sorted(second)})
    zero, _ = random_deleted([3, 1, 2], 0, 1, 2)
    require(zero == set(), "SELF_TEST_ZERO_FAIL", {"actual": sorted(zero)})
    zero_parent = sorted(parent_deleted(rows, {"absent_host"}, {"absent_guest"}))
    require(zero_parent == [], "SELF_TEST_ZERO_PARENT_FAIL", {"actual": zero_parent})
    boundary = {"49": int(49 >= 50), "50": int(50 >= 50), "51": int(51 >= 50)}
    require(boundary == {"49": 0, "50": 1, "51": 1}, "SELF_TEST_MIN_TRAIN_FAIL", boundary)
    report = {"status": "PASS", "or_deleted": actual_or, "random_seed": seed,
              "random_deleted_from_sorted_1_2_4_6_9": sorted(first), "zero_deleted": sorted(zero),
              "zero_parent_deleted": zero_parent,
              "minimum_train_50_boundary": boundary}
    print(json.dumps({"self_test": report}, ensure_ascii=False))
    return report


def input_paths(args):
    paths = {}
    for name, filename in INPUT_NAMES.items():
        supplied = getattr(args, name)
        if supplied is None:
            raise ValueError("Normal execution requires --" + name.replace("_", "-"))
        path = pathlib.Path(supplied).resolve()
        if path != (IDENTITY / filename).resolve():
            raise ValueError("Input is outside the exact identity whitelist: " + name)
        paths[name] = path
    if args.recipes is None or pathlib.Path(args.recipes).resolve() != (ROOT / "inputs" / "frozen_recipes.json").resolve():
        raise ValueError("--recipes must name inputs/frozen_recipes.json")
    if args.output_dir is None or pathlib.Path(args.output_dir).resolve() != (ROOT / "masks").resolve():
        raise ValueError("--output-dir must be the masks directory")
    output_dir = pathlib.Path(args.output_dir).resolve()
    for filename in (*OUTPUTS, CONFLICT_FILE):
        if (output_dir / filename).exists():
            raise ValueError("Refusing an existing concrete output: " + str(output_dir / filename))
    return paths, output_dir


def load_identity(paths, recipes_path):
    source = {name: read_csv(path, name) for name, path in paths.items()}
    recipes = json.loads(pathlib.Path(recipes_path).read_text(encoding="utf-8-sig"))
    definition = recipes["double_cold_definition"]
    require(definition["split"] == "double_cold" and definition["fold_index"] == "seed",
            "RECIPE_IDENTITY_CONFLICT", {"split": definition["split"], "fold_index": definition["fold_index"]})
    minimum_train = int(definition["validity"]["minimum_train_rows"])
    minimum_test = int(definition["validity"]["minimum_test_rows"])
    require(minimum_train == 50 and minimum_test == 25, "RECIPE_MINIMUM_CONFLICT",
            {"minimum_train": minimum_train, "minimum_test": minimum_test})
    aliases = {}
    for r in source["aliases"]:
        role, raw_id = r["entity_type"], int(r["raw_id"])
        require(role in ("host", "guest"), "ALIAS_ROLE_CONFLICT", r)
        put_unique(aliases, (role, raw_id), sha(r["canonical_key_sha256"], {"role": role, "raw_id": raw_id}),
                   "RAW_ID_CANONICAL_ALIAS_CONFLICT", {})
    mapping = {}
    for r in source["parent_mapping"]:
        role, raw_id = r["role"], int(r["identity_id"])
        require(role in ("host", "guest"), "PARENT_MAPPING_ROLE_CONFLICT", r)
        put_unique(mapping, (role, raw_id), r, "RAW_ID_PARENT_MAPPING_CONFLICT", {})
    require(set(mapping) == set(aliases), "MAPPING_ALIAS_COVERAGE_CONFLICT",
            {"mapping_only": sorted(set(mapping) - set(aliases)), "alias_only": sorted(set(aliases) - set(mapping))})
    parents, uncertain = {}, {}
    nonempty_sha_checks = 0
    for r in source["appearance_keys"]:
        require(r["layer"] == "v2_reviewed_only_parent", "PARENT_LAYER_CONFLICT", {"actual": r["layer"]})
        flags = {}
        for role in ("host", "guest"):
            raw_id = int(r[role + "_id"])
            key = (role, raw_id)
            require(key in aliases, "APPEARANCE_ALIAS_MISSING", {"role": role, "raw_id": raw_id})
            parent = r["parent_" + role + "_key"]
            require(bool(parent), "EMPTY_PARENT_KEY", {"role": role, "raw_id": raw_id})
            put_unique(parents, key, parent, "RAW_ID_PRIMARY_PARENT_CONFLICT", {})
            flags[role] = boolean(r[role + "_uncertain"])
            put_unique(uncertain, key, flags[role], "RAW_ID_UNCERTAIN_FLAG_CONFLICT", {})
            canonical = r["canonical_" + role + "_sha256"]
            if canonical:
                require(sha(canonical, {"role": role, "raw_id": raw_id}) == aliases[key],
                        "APPEARANCE_CANONICAL_ALIAS_CONFLICT", {"role": role, "raw_id": raw_id,
                         "appearance": canonical, "alias": aliases[key]})
                nonempty_sha_checks += 1
        require(boolean(r["uncertain"]) == (flags["host"] or flags["guest"]),
                "APPEARANCE_COMBINED_UNCERTAIN_CONFLICT", {"host_id": r["host_id"], "guest_id": r["guest_id"]})
    require(set(parents) == set(aliases), "PRIMARY_PARENT_COVERAGE_CONFLICT",
            {"missing": sorted(set(aliases) - set(parents)), "unexpected": sorted(set(parents) - set(aliases))})
    for key, row in mapping.items():
        require(parents[key] in (row["strict_exact_key"], row["parent_neutralized_operational_key"]),
                "PRIMARY_NOT_STRICT_OR_FROZEN_OPERATIONAL", {"role": key[0], "raw_id": key[1]})
    decision_ids = set()
    decisions = {}
    decision_counts = collections.Counter()
    for r in source["decisions"]:
        require(r["review_group_id"] not in decision_ids, "DUPLICATE_REVIEW_GROUP_ID", r)
        decision_ids.add(r["review_group_id"])
        decisions[r["review_group_id"]] = r["decision"]
        decision_counts[r["decision"]] += 1
    require(dict(decision_counts) == {"accepted_as_operational": 44, "uncertain": 3},
            "PARENT_DECISION_COUNTS_CONFLICT", dict(decision_counts))
    uncertain_links = {}
    for r in source["uncertain_links"]:
        key = (r["role"], int(r["raw_id"]))
        group_id = r["review_group_id"]
        require(key in mapping, "UNCERTAIN_LINK_IDENTITY_MISSING", {"role": key[0], "raw_id": key[1]})
        require(group_id in decisions and decisions[group_id] == "uncertain", "UNCERTAIN_LINK_DECISION_CONFLICT", r)
        require(group_id.startswith("v2_parent_neutralized:" + key[0] + ":"), "UNCERTAIN_LINK_AXIS_CONFLICT", r)
        put_unique(uncertain_links, key, group_id, "UNCERTAIN_LINK_GROUP_CONFLICT", {})
    require(len(uncertain_links) == 7 and set(uncertain_links.values()) == {g for g, d in decisions.items() if d == "uncertain"},
            "UNCERTAIN_LINK_COVERAGE_CONFLICT", {"member_count": len(uncertain_links), "actual_groups": sorted(set(uncertain_links.values()))})
    groups = collections.defaultdict(list)
    for key, row in mapping.items():
        groups[(key[0], row["parent_neutralized_operational_key"])].append(key)
    fallback_groups = []
    anchor_groups = set()
    fallback_members = set()
    for role, anchor in (("host", 30), ("host", 82), ("guest", 856)):
        anchor_key = (role, anchor)
        require(anchor_key in mapping, "FALLBACK_ANCHOR_MISSING", {"role": role, "raw_id": anchor})
        op = mapping[anchor_key]["parent_neutralized_operational_key"]
        require((role, op) not in anchor_groups, "FALLBACK_ANCHOR_GROUPS_NOT_DISTINCT", {"role": role, "raw_id": anchor})
        anchor_groups.add((role, op))
        members = sorted(groups[(role, op)])
        require(anchor_key in uncertain_links, "FALLBACK_ANCHOR_REVIEW_LINK_MISSING", {"role": role, "raw_id": anchor})
        group_id = uncertain_links[anchor_key]
        require(len(members) > 1, "FALLBACK_GROUP_MISSING_MEMBERS", {"role": role, "anchor": anchor})
        actual = []
        for key in members:
            require(uncertain_links.get(key) == group_id, "FALLBACK_GROUP_MEMBER_REVIEW_LINK_CONFLICT",
                    {"role": key[0], "raw_id": key[1], "actual": uncertain_links.get(key), "expected": group_id})
            fallback_members.add(key)
            primary_is_strict = parents[key] == mapping[key]["strict_exact_key"]
            require(primary_is_strict and uncertain[key], "FALLBACK_MEMBER_NOT_STRICT_UNCERTAIN",
                    {"role": key[0], "raw_id": key[1], "primary_is_strict": primary_is_strict, "uncertain": uncertain[key]})
            actual.append({"raw_id": key[1], "primary_is_strict": True,
                           "review_group_id": uncertain_links[key],
                           "strict_equals_operational": mapping[key]["strict_exact_key"] == op,
                           "uncertain": True})
        fallback_groups.append({"role": role, "anchor_raw_id": anchor, "review_group_id": group_id,
                                "same_operational_key_members": actual})
    require(fallback_members == set(uncertain_links), "FALLBACK_MEMBER_COVERAGE_CONFLICT",
            {"link_only": sorted(set(uncertain_links) - fallback_members), "group_only": sorted(fallback_members - set(uncertain_links))})
    blocks = collections.defaultdict(list)
    row_identity = {}
    role_coverage = collections.defaultdict(set)
    membership_sha_checks = 0
    for r in source["membership"]:
        require(r["split"] == "double_cold", "MEMBERSHIP_SPLIT_CONFLICT", {"actual": r["split"]})
        seed, fold = int(r["seed"]), int(r["fold_index"])
        require(seed == fold, "MEMBERSHIP_SEED_FOLD_CONFLICT", {"seed": seed, "fold_index": fold})
        require(r["role"] in ("train", "test", "neither"), "MEMBERSHIP_ROLE_CONFLICT", {"actual": r["role"]})
        row = {"row_index": int(r["row_index"]), "original_role": r["role"],
               "host_id": int(r["host_id"]), "guest_id": int(r["guest_id"])}
        for role in ("host", "guest"):
            raw_id = row[role + "_id"]
            key = (role, raw_id)
            require(key in aliases and key in parents, "MEMBERSHIP_IDENTITY_MISSING", {"role": role, "raw_id": raw_id})
            canonical = sha(r["canonical_" + role + "_sha256"], {"role": role, "raw_id": raw_id})
            require(canonical == aliases[key], "MEMBERSHIP_CANONICAL_ALIAS_CONFLICT",
                    {"role": role, "raw_id": raw_id, "membership": canonical, "alias": aliases[key]})
            membership_sha_checks += 1
            row["canonical_" + role + "_sha256"] = canonical
            row["parent_" + role + "_key"] = parents[key]
            row[role + "_uncertain"] = uncertain[key]
            role_coverage[r["role"]].add(key)
        identity = tuple(row[k] for k in ("host_id", "guest_id", "canonical_host_sha256", "canonical_guest_sha256"))
        if row["row_index"] in row_identity:
            require(row_identity[row["row_index"]] == identity, "ROW_IDENTITY_CHANGED_ACROSS_BLOCKS", {"row_index": row["row_index"]})
        row_identity[row["row_index"]] = identity
        blocks[(seed, fold)].append(row)
    expected_seeds = [int(v) for v in definition["attempted_seeds"]]
    require(sorted(blocks) == [(s, s) for s in expected_seeds] == [(s, s) for s in range(12)],
            "BLOCK_SET_CONFLICT", {"actual": sorted(blocks), "expected_seeds": expected_seeds})
    require(len(row_identity) == 2383, "ROW_UNIVERSE_COUNT_CONFLICT", {"actual": len(row_identity), "expected": 2383})
    universe = set(row_identity)
    for block, rows in blocks.items():
        indices = [r["row_index"] for r in rows]
        require(len(indices) == 2383 and len(set(indices)) == 2383 and set(indices) == universe,
                "BLOCK_ROW_UNIVERSE_CONFLICT", {"seed": block[0], "fold_index": block[1], "rows": len(indices), "unique": len(set(indices))})
        rows.sort(key=lambda r: r["row_index"])
    total_tests = sum(r["original_role"] == "test" for rows in blocks.values() for r in rows)
    require(total_tests == 1716, "ORIGINAL_TEST_APPEARANCE_CONFLICT", {"actual": total_tests, "expected": 1716})
    coverage = {
        "status": "IDENTITY_ONLY_MASKS_BUILT", "membership_input_rows": len(source["membership"]),
        "original_row_universe": len(universe), "original_test_appearances": total_tests,
        "block_count": len(blocks), "source_projection_rows": {k: len(v) for k, v in source.items()},
        "identity_counts": dict(collections.Counter(k[0] for k in aliases)),
        "raw_id_primary_parent_conflicts": 0, "raw_id_canonical_alias_conflicts": 0,
        "nonempty_appearance_sha_alias_checks": nonempty_sha_checks, "membership_sha_alias_checks": membership_sha_checks,
        "all_membership_roles_identity_covered": True,
        "role_identity_counts": {role: dict(collections.Counter(k[0] for k in keys)) for role, keys in role_coverage.items()},
        "decisions": dict(decision_counts), "fallback_groups": fallback_groups,
        "review_group_id_linkage": {
            "status": "COMPLETE_LITERAL_FROZEN_GROUP_REFERENCE_JOIN",
            "member_count": len(uncertain_links), "group_count": len(set(uncertain_links.values())),
            "evidence": "The v2 links join all seven members to the three uncertain frozen decisions; same-operational-key groups agree and every member retains its own strict key. No hash rule is inferred for review_group_id.",
        },
        "canonical_to_parent_global_uniqueness_enforced": False,
        "row_index_semantics": "zero-based physical clean_cond row index; never parquet_row",
        "boundary": {
            "opened_data_inputs": [str(paths[k]) for k in INPUT_NAMES],
            "opened_metadata_input": str(pathlib.Path(recipes_path).resolve()),
            "targets_opened": False, "npz_opened": False, "predictions_opened": False,
            "outcome_tables_opened": False, "reported_results_opened": False, "chemical_reprocessing": False,
            "target_validity_gate": "DEFERRED_TO_STAGE_B",
        },
    }
    return blocks, minimum_train, minimum_test, coverage


def construct(blocks, minimum_train, minimum_test):
    block_map, support, plans = [], [], []
    for b, ((seed, fold), rows) in enumerate(sorted(blocks.items())):
        block_id = "double_cold/seed_" + str(fold)
        common = {"b": b, "block_id": block_id, "seed": seed, "fold_index": fold}
        block_map.append(common)
        train = [r for r in rows if r["original_role"] == "train"]
        test = [r for r in rows if r["original_role"] == "test"]
        neither = [r for r in rows if r["original_role"] == "neither"]
        require(len(train) >= minimum_train and len(test) >= minimum_test, "ORIGINAL_STRUCTURAL_GATE_FAIL",
                {**common, "train": len(train), "test": len(test)})
        test_hosts = {r["parent_host_key"] for r in test}
        test_guests = {r["parent_guest_key"] for r in test}
        test_raw_hosts = {r["host_id"] for r in test}
        test_raw_guests = {r["guest_id"] for r in test}
        for role in ("host", "guest"):
            train_keys = {r["canonical_" + role + "_sha256"] for r in train}
            test_keys = {r["canonical_" + role + "_sha256"] for r in test}
            require(not train_keys.intersection(test_keys), "ORIGINAL_CANONICAL_SAME_AXIS_OVERLAP",
                    {**common, "role": role, "intersection_count": len(train_keys.intersection(test_keys))})
        purge = parent_deleted(train, test_hosts, test_guests)
        arm_plans = []
        for arm in ARMS:
            mask_seed = ""
            if arm == "baseline":
                deleted = set()
            elif arm == "parent_purge":
                deleted = purge
            else:
                deleted, mask_seed = random_deleted([r["row_index"] for r in train], len(purge), b, int(arm[-1]))
            retained = [r for r in train if r["row_index"] not in deleted]
            hosts = {r["parent_host_key"] for r in retained}
            guests = {r["parent_guest_key"] for r in retained}
            host_intersection = hosts.intersection(test_hosts)
            guest_intersection = guests.intersection(test_guests)
            require(len(retained) >= minimum_train, "ARM_MINIMUM_TRAIN_FAIL",
                    {**common, "arm": arm, "retained_train": len(retained), "minimum_train": minimum_train})
            if arm == "parent_purge":
                require(not host_intersection and not guest_intersection, "PARENT_PURGE_RESIDUAL_INTERSECTION",
                        {**common, "host_count": len(host_intersection), "guest_count": len(guest_intersection)})
            if arm.startswith("random_purge"):
                require(len(deleted) == len(purge), "RANDOM_DELETE_COUNT_CONFLICT", {**common, "arm": arm})
            removed = [r for r in train if r["row_index"] in deleted]
            either_uncertain = lambda r: r["host_uncertain"] or r["guest_uncertain"]
            support.append({
                **common, "arm": arm, "original_train_rows": len(train), "original_test_rows": len(test),
                "original_neither_rows": len(neither), "retained_train_rows": len(retained),
                "deleted_train_rows": len(deleted), "deleted_train_fraction": len(deleted) / len(train),
                "remaining_unique_raw_hosts": len({r["host_id"] for r in retained}),
                "remaining_unique_raw_guests": len({r["guest_id"] for r in retained}),
                "remaining_unique_parent_hosts": len(hosts), "remaining_unique_parent_guests": len(guests),
                "remaining_host_parent_exposure_rows": sum(r["parent_host_key"] in test_hosts for r in retained),
                "remaining_guest_parent_exposure_rows": sum(r["parent_guest_key"] in test_guests for r in retained),
                "remaining_any_parent_exposure_rows": sum(r["parent_host_key"] in test_hosts or r["parent_guest_key"] in test_guests for r in retained),
                "remaining_host_parent_test_intersection_count": len(host_intersection),
                "remaining_guest_parent_test_intersection_count": len(guest_intersection),
                "remaining_raw_host_test_intersection_count": len({r["host_id"] for r in retained}.intersection(test_raw_hosts)),
                "remaining_raw_guest_test_intersection_count": len({r["guest_id"] for r in retained}.intersection(test_raw_guests)),
                "mask_seed": mask_seed, "minimum_train_rows": minimum_train, "min_train_pass": int(len(retained) >= minimum_train),
                "minimum_test_rows": minimum_test, "min_test_pass": int(len(test) >= minimum_test),
                "original_train_uncertain_rows": sum(either_uncertain(r) for r in train),
                "deleted_train_uncertain_rows": sum(either_uncertain(r) for r in removed),
                "retained_train_uncertain_rows": sum(either_uncertain(r) for r in retained),
                "deleted_train_host_uncertain_rows": sum(r["host_uncertain"] for r in removed),
                "deleted_train_guest_uncertain_rows": sum(r["guest_uncertain"] for r in removed),
                "retained_train_host_uncertain_rows": sum(r["host_uncertain"] for r in retained),
                "retained_train_guest_uncertain_rows": sum(r["guest_uncertain"] for r in retained),
            })
            arm_plans.append((arm, deleted, mask_seed))
        plans.append((common, rows, test_hosts, test_guests, arm_plans))
    return block_map, support, plans


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mask_rows(plans):
    for common, rows, test_hosts, test_guests, arm_plans in plans:
        for arm, deleted, mask_seed in arm_plans:
            for row in rows:
                is_train = row["original_role"] == "train"
                yield {
                    **common, **row, "arm": arm,
                    "retained_train": int(is_train and row["row_index"] not in deleted),
                    "deleted_train": int(is_train and row["row_index"] in deleted),
                    "host_uncertain": int(row["host_uncertain"]), "guest_uncertain": int(row["guest_uncertain"]),
                    "host_parent_seen_in_test": int(row["parent_host_key"] in test_hosts),
                    "guest_parent_seen_in_test": int(row["parent_guest_key"] in test_guests), "mask_seed": mask_seed,
                }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), dest=name)
    parser.add_argument("--recipes")
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    self_test_report = self_test()
    if args.self_test:
        if any(getattr(args, name) is not None for name in (*INPUT_NAMES, "recipes", "output_dir")):
            parser.error("--self-test does not accept real input or output arguments")
        return 0
    paths, output_dir = input_paths(args)
    try:
        blocks, minimum_train, minimum_test, coverage = load_identity(paths, args.recipes)
        block_map, support, plans = construct(blocks, minimum_train, minimum_test)
    except IdentityConflict as exc:
        output_dir.mkdir(exist_ok=True)
        report = {"status": "SOURCE_CONFLICT", "code": exc.code, "details": exc.details,
                  "masks_generated": False, "self_test": self_test_report}
        with (output_dir / CONFLICT_FILE).open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(report, ensure_ascii=False))
        return 2
    coverage["self_test"] = self_test_report
    coverage["output_rows"] = {"train_masks.csv": 12 * 5 * 2383, "block_map.csv": 12, "block_arm_support.csv": 60}
    coverage["parent_purge_zero_same_axis_parent_intersection_all_blocks"] = True
    coverage["all_arms_minimum_train_50_pass"] = True
    coverage["all_blocks_minimum_test_25_pass"] = True
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "block_map.csv", ["b", "block_id", "seed", "fold_index"], block_map)
    write_csv(output_dir / "block_arm_support.csv", SUPPORT_COLUMNS, support)
    write_csv(output_dir / "train_masks.csv", MASK_COLUMNS, mask_rows(plans))
    with (output_dir / "identity_coverage.json").open("x", encoding="utf-8") as handle:
        json.dump(coverage, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": coverage["status"], "output_dir": str(output_dir),
                      "rows": coverage["output_rows"], "parent_support": [
                          {k: r[k] for k in ("b", "block_id", "original_train_rows", "original_test_rows", "deleted_train_rows", "retained_train_rows")}
                          for r in support if r["arm"] == "parent_purge"],
                      "minimum_retained_train": min(r["retained_train_rows"] for r in support)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IdentityConflict as exc:
        print(json.dumps({"status": "STOP", "code": exc.code, "actual_details": exc.details}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "STOP", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
