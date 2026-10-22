"""Location adapter for the public export of the frozen Stage A/B analysis.

The public freeze binds exported contract/script bytes; its source identity,
path changes and label, comment and message wording changes are recorded in
../PUBLIC_EXPORT_MANIFEST.json. Scientific inputs, masks, model settings and
saved results retain their identities; the E1/E2 runner code is unchanged
apart from comments and messages.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


class BindingError(ValueError):
    pass


class Adapter:
    def __init__(self, companion_root: Path, snapshot_root: Path):
        self.root = Path(companion_root).resolve()
        self.original = self.root / "original_run"
        self.snapshot = Path(snapshot_root).resolve()
        self._identities = {}
        self.original_contract = json.loads((self.original / "analysis_contract.json").read_text(encoding="utf-8"))
        self.freeze = json.loads((self.original / "stage_a_freeze.json").read_text(encoding="utf-8"))
        self.registry = json.loads((self.root / "input_sources/SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
        if self.registry["snapshot_commit"] != self.original_contract["scientific_snapshot"]["commit"]:
            raise BindingError("source registry snapshot identity mismatch")
        if set(self.registry["sources"]) != set(self.original_contract["sources"]):
            raise BindingError("source registry must contain exactly the frozen source IDs")
        if self.freeze.get("public_export", {}).get("schema") != "suprabench-public-freeze-v1":
            raise BindingError("the public export freeze is required")
        if self.original_contract.get("public_path_binding", {}).get("schema") != "suprabench-public-paths-v1":
            raise BindingError("the public relative-path contract is required")
        if self.original_contract["experiment_root"] != ".":
            raise BindingError("public experiment_root must denote original_run")
        for sid, saved in self.original_contract["sources"].items():
            mapping = self.registry["sources"][sid]
            if any(mapping[k] != saved[k] for k in ("bytes", "sha256")):
                raise BindingError("registry identity differs from frozen source: " + sid)
            if "archive_member" in saved:
                prefix = "suprabench-cold-split-audit-" + self.registry["snapshot_commit"] + "/"
                if not saved["archive_member"].startswith(prefix):
                    raise BindingError("unexpected frozen archive prefix")
                expected = {"scope": "snapshot", "relative_path": saved["archive_member"][len(prefix):]}
            else:
                expected = {"scope": "original_run", "relative_path": saved["path"]}
            if saved["path"] != expected["relative_path"]:
                raise BindingError("public contract source location mismatch: " + sid)
            if any(mapping[k] != expected[k] for k in expected):
                raise BindingError("registry location differs from frozen source: " + sid)
        self.contract = copy.deepcopy(self.original_contract)
        self.contract["experiment_root"] = str(self.original)
        for sid in self.contract["sources"]:
            self.contract["sources"][sid]["path"] = str(self.source_path(sid))

    def file_identity(self, path: Path) -> dict:
        path = Path(path).resolve()
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        cached = self._identities.get(path)
        if cached is not None:
            if cached[0] != fingerprint:
                raise BindingError("bound file changed during this invocation: " + str(path))
            return dict(cached[1])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity = {"bytes": stat.st_size, "sha256": digest.hexdigest()}
        self._identities[path] = (fingerprint, identity)
        return dict(identity)

    def source_path(self, source_id: str) -> Path:
        if source_id not in self.registry["sources"]:
            raise BindingError("unknown frozen source ID: " + source_id)
        mapping = self.registry["sources"][source_id]
        if mapping["scope"] not in {"snapshot", "original_run"}:
            raise BindingError("unknown source root scope: " + source_id)
        base = self.snapshot if mapping["scope"] == "snapshot" else self.original
        if Path(mapping["relative_path"]).is_absolute():
            raise BindingError("source location must be relative: " + source_id)
        path = (base / mapping["relative_path"]).resolve()
        if not path.is_relative_to(base):
            raise BindingError("source escapes its declared root")
        return path

    def verify_source(self, contract: dict, source_id: str, path: Path) -> None:
        expected_path = self.source_path(source_id)
        if Path(path).resolve() != expected_path:
            raise BindingError("source path differs from its mapped frozen location: " + source_id)
        expected = self.original_contract["sources"][source_id]
        actual = self.file_identity(expected_path)
        if any(actual[k] != expected[k] for k in ("bytes", "sha256")):
            raise BindingError("source bytes differ from frozen identity: " + source_id)

    def load_and_verify_freeze(self, contract_path: Path, receipt_path: Path) -> dict:
        if Path(contract_path).resolve() != self.original / "analysis_contract.json" or Path(receipt_path).resolve() != self.original / "stage_a_freeze.json":
            raise BindingError("only the public contract and public freeze receipt are accepted")
        if self.freeze["status"] != "STAGE_A_FROZEN_READY_FOR_SEPARATE_STAGE_B":
            raise BindingError("the preserved Stage A receipt is not frozen")
        if self.original_contract["schema_version"] != "e1e2-analysis-contract-v1":
            raise BindingError("unsupported original contract schema")
        for relative, expected in self.freeze["frozen_files"].items():
            if expected["kind"] not in {"contract", "script"}:
                continue
            path = (self.original / relative).resolve()
            if not path.is_relative_to(self.original):
                raise BindingError("frozen member escapes the preserved run")
            actual = self.file_identity(path)
            if any(actual[k] != expected[k] for k in ("bytes", "sha256")):
                raise BindingError("frozen executable/contract drift: " + relative)
        for relative, expected in self.freeze["public_export"]["adapter_files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise BindingError("adapter member escapes the public companion")
            actual = self.file_identity(path)
            if any(actual[k] != expected[k] for k in ("bytes", "sha256")):
                raise BindingError("public adapter/registry drift: " + relative)
        return self.contract

    def new_output_dir(self, contract: dict, path: Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            raise BindingError("output directory must be explicit and absolute")
        path = path.resolve()
        if path.is_relative_to(self.root) or path.is_relative_to(self.snapshot):
            raise BindingError("new output must be outside the preserved companion and snapshot")
        if path.exists():
            raise FileExistsError("refusing an existing output directory: " + str(path))
        path.mkdir(parents=True, exist_ok=False)
        return path


def write_json(path: Path, payload: object) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
