"""Small shared file guards for the E1/E2 scripts; no import-time I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def file_identity(path: Path) -> dict:
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b''):
            h.update(chunk)
    return {'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def write_json(path: Path, payload: object) -> None:
    with Path(path).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(payload,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def load_and_verify_freeze(contract_path: Path, receipt_path: Path) -> dict:
    root=Path(__file__).resolve().parents[1]
    contract_path=Path(contract_path).resolve()
    receipt_path=Path(receipt_path).resolve()
    if contract_path != root/'analysis_contract.json' or receipt_path != root/'stage_a_freeze.json':
        raise ValueError('only the frozen contract and stage-A freeze receipt are accepted')
    with receipt_path.open(encoding='utf-8') as handle:
        receipt=json.load(handle)
    if receipt['status']!='STAGE_A_FROZEN_READY_FOR_SEPARATE_STAGE_B':
        raise ValueError('stage A has not passed and frozen')
    # Masks and real inputs are checked by verify_source immediately before use.
    # Check the contract and every executable here without rechecking unrelated artifacts.
    for relative,expected in receipt['frozen_files'].items():
        if expected['kind'] not in {'contract','script'}:
            continue
        path=(root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('freeze member escapes the experiment root')
        actual=file_identity(path)
        if any(actual[key]!=expected[key] for key in ('bytes','sha256')):
            raise ValueError('frozen executable/contract drift: '+relative)
    with contract_path.open(encoding='utf-8') as handle:
        contract=json.load(handle)
    if Path(contract['experiment_root']).resolve()!=root:
        raise ValueError('contract experiment root mismatch')
    if contract['schema_version']!='e1e2-analysis-contract-v1':
        raise ValueError('unsupported contract schema')
    return contract


def verify_source(contract: dict, source_id: str, path: Path) -> None:
    expected=contract['sources'][source_id]
    path=Path(path).resolve()
    if path!=Path(expected['path']).resolve():
        raise ValueError('source path differs from the frozen contract: '+source_id)
    actual=file_identity(path)
    if any(actual[key]!=expected[key] for key in ('bytes','sha256')):
        raise ValueError('source bytes differ from the frozen contract: '+source_id)


def new_output_dir(contract: dict, path: Path) -> Path:
    path=Path(path)
    if not path.is_absolute():
        raise ValueError('output directory must be an explicit absolute path')
    root=Path(contract['experiment_root']).resolve()
    allowed=(root/'results').resolve()
    path=path.resolve()
    if not path.is_relative_to(allowed) or path==allowed:
        raise ValueError('output must be a new child directory of this experiment/results')
    if path.exists():
        raise FileExistsError('refusing to overwrite an existing output directory: '+str(path))
    path.mkdir(parents=True,exist_ok=False)
    return path
