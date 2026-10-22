"""Stage A: project frozen identity/membership columns, never targets or predictions."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

SPECS = {
    'canonical_membership': ('data/canonical_cpu/folds/folds_double_canonical.csv', ',', ['split','fold_index','seed','role','row_index','host_id','guest_id','canonical_host_sha256','canonical_guest_sha256']),
    'parent_appearance_keys': ('results/parent_identity_b1/stage_b/APPEARANCE_CLASSIFICATION_MASKS.csv', ',', ['layer','host_id','guest_id','canonical_host_sha256','canonical_guest_sha256','parent_host_key','parent_guest_key','host_uncertain','guest_uncertain','uncertain']),
    'canonical_aliases': ('data/identity/canonical_identity_alias_map_2026-08-01.tsv', '\t', ['entity_type','raw_id','canonical_key_sha256']),
    'parent_mapping': ('results/parent_identity_b1/identity/v2_identity_mapping.csv', ',', ['role','identity_id','strict_exact_key','parent_neutralized_operational_key','uncertain_reasons']),
    'parent_decisions': ('results/parent_identity_b1/identity/v2_group_review_decisions.csv', ',', ['review_group_id','decision']),
}
CONFIGS = {
    'canonical_config': 'data/canonical_cpu/config/canonical_rebuild_config.json',
    'model_config': 'data/canonical_cpu/config/model_stage_config.json',
}
FORBIDDEN = {'y','y_true','y_pred','y_true_serialized','r2','mae','spearman'}

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot-zip', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.output_dir.resolve()
    if out != root/'inputs'/'identity' or any(out.iterdir()):
        raise RuntimeError('identity output must be the empty inputs/identity directory')
    source_root=args.source_root.resolve()
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(), 'snapshot_zip':str(args.snapshot_zip.resolve()),
             'source_root':str(source_root), 'stage':'A_IDENTITY_ONLY', 'sources':{}, 'projections':{},
             'excluded_content':'No NPZ, targets, predictions or outcome tables are opened.'}
    with ZipFile(args.snapshot_zip) as z:
        prefix='suprabench-cold-split-audit-81b1d092b4d2bdd4a4f293f340e7d369395ac053/'
        def read_bound(source_id: str, rel: str) -> bytes:
            payload=z.read(prefix+rel)
            live=source_root/rel
            if live.read_bytes()!=payload:
                raise RuntimeError('fixed snapshot/local source mismatch: '+rel)
            receipt['sources'][source_id]={'path':str(live),'archive_member':prefix+rel,'bytes':len(payload),'sha256':digest(payload)}
            return payload
        for source_id,(rel,delimiter,columns) in SPECS.items():
            if set(columns)&FORBIDDEN:
                raise RuntimeError('forbidden projection column')
            payload=read_bound(source_id,rel)
            reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig'),newline=''),delimiter=delimiter)
            if not set(columns).issubset(reader.fieldnames or []):
                raise RuntimeError('missing input columns: '+source_id)
            rows=[]; source_rows=0
            for row in reader:
                source_rows+=1
                if source_id=='parent_appearance_keys' and row['layer']!='v2_reviewed_only_parent':
                    continue
                rows.append(tuple(row[c] for c in columns))
            if source_id=='parent_appearance_keys':
                rows=sorted(set(rows),key=lambda x:(int(x[1]),int(x[2]),x))
            output=out/(source_id+'.csv')
            with output.open('x',encoding='utf-8',newline='') as f:
                writer=csv.writer(f,lineterminator='\n');writer.writerow(columns);writer.writerows(rows)
            output_bytes=output.read_bytes()
            receipt['projections'][source_id]={'path':str(output),'columns':columns,'rows':len(rows),'source_rows':source_rows,
                                                'bytes':len(output_bytes),'sha256':digest(output_bytes),
                                                'selection':'all appearances, primary layer, exact projected-tuple deduplication' if source_id=='parent_appearance_keys' else 'all rows, specified columns only'}
        canonical=json.loads(read_bound('canonical_config',CONFIGS['canonical_config']))
        model=json.loads(read_bound('model_config',CONFIGS['model_config']))
        recipes={'identity_policy':canonical['identity_policy'],
                 'double_cold_definition':canonical['fold_definitions']['double_cold'],
                 'rf':model['model_recipes']['rf'],'additive':model['model_recipes']['additive'],
                 'sim_knn_k5':model['model_recipes']['sim_knn_k5']}
        save_json(root/'inputs'/'frozen_recipes.json',recipes)
    save_json(root/'inputs'/'projection_receipt.json',receipt)
    print(json.dumps({'status':'IDENTITY_PROJECTION_COMPLETE','projections':{k:v['rows'] for k,v in receipt['projections'].items()},
                      'receipt':str(root/'inputs'/'projection_receipt.json')},ensure_ascii=False))

if __name__=='__main__':
    main()
