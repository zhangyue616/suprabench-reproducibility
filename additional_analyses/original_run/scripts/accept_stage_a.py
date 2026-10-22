"""One consolidated Stage A acceptance; synthetic models and identity-only samples."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from staged_io import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / 'acceptance' / 'stage_a_acceptance.json'
    if destination.exists():
        raise FileExistsError(destination)
    synthetic = {}
    for script in ('e2_rank_diagnostic.py', 'e1_parent_purge_runner.py'):
        process = subprocess.run([sys.executable, '-B', str(ROOT/'scripts'/script), '--self-test'],
                                 text=True, encoding='utf-8', capture_output=True)
        result = {'exit_code': process.returncode, 'stderr': process.stderr,
                  'stdout': json.loads(process.stdout) if process.stdout else None}
        write_json(ROOT/'acceptance'/(script.removesuffix('.py')+'_selftest.json'), result)
        if process.returncode:
            raise RuntimeError('synthetic self-test failed: '+script)
        synthetic[script] = result

    # Sample already constructed masks; they are not regenerated here.
    samples = {0: [], 4: [], 11: []}
    with (ROOT/'masks/train_masks.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            b = int(row['b'])
            if b in samples:
                samples[b].append(row)
    with (ROOT/'inputs/identity/canonical_membership.csv').open(encoding='utf-8', newline='') as handle:
        original = {(int(r['seed']), int(r['row_index'])): r for r in csv.DictReader(handle)
                    if int(r['seed']) in samples}
    reports = []
    for b, rows in samples.items():
        baseline = {int(r['row_index']): r for r in rows if r['arm']=='baseline'}
        assert len(baseline)==2383
        held = [r for r in baseline.values() if r['original_role']=='test']
        h = {r['parent_host_key'] for r in held}
        g = {r['parent_guest_key'] for r in held}
        train = sorted(i for i,r in baseline.items() if r['original_role']=='train')
        expected_parent = {i for i in train if baseline[i]['parent_host_key'] in h
                           or baseline[i]['parent_guest_key'] in g}
        actual_parent = {int(r['row_index']) for r in rows
                         if r['arm']=='parent_purge' and r['deleted_train']=='1'}
        assert actual_parent==expected_parent
        j = (0, 1, 2)[(0, 4, 11).index(b)]
        seed = 100000+100*b+j
        expected_random = set(np.random.RandomState(seed).choice(
            np.asarray(train,dtype=np.int64), size=len(expected_parent), replace=False).tolist())
        random_rows = [r for r in rows if r['arm']==f'random_purge_{j}']
        actual_random = {int(r['row_index']) for r in random_rows if r['deleted_train']=='1'}
        assert expected_random==actual_random
        assert {r['mask_seed'] for r in random_rows}=={str(seed)}
        for row in rows:
            source = original[(b,int(row['row_index']))]
            assert row['original_role']==source['role']
            assert all(row[k]==source[k] for k in
                       ('host_id','guest_id','canonical_host_sha256','canonical_guest_sha256'))
            if row['original_role']!='train':
                assert row['retained_train']==row['deleted_train']=='0'
        reports.append({'b':b, 'parent_deleted':len(actual_parent), 'random_j':j,
                        'random_seed':seed, 'status':'PASS',
                        'checks':['original role/identity and test/neither preservation',
                                  'same-axis OR parent purge', 'sorted RandomState matched deletion']})
    coverage = json.loads((ROOT/'masks/identity_coverage.json').read_text(encoding='utf-8'))
    with (ROOT/'masks/block_arm_support.csv').open(encoding='utf-8',newline='') as handle:
        support = list(csv.DictReader(handle))
    assert len(support)==60
    assert all(r['min_train_pass']==r['min_test_pass']=='1' for r in support)
    parent = [r for r in support if r['arm']=='parent_purge']
    assert len(parent)==12
    assert all(r['remaining_host_parent_test_intersection_count']==
               r['remaining_guest_parent_test_intersection_count']=='0' for r in parent)
    write_json(destination, {
        'status':'STAGE_A_IMPLEMENTATION_ACCEPTED', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'synthetic_self_tests':synthetic, 'independent_mask_samples':reports,
        'reused_constructor_evidence':{'path':'masks/identity_coverage.json',
            'identity_counts':coverage['identity_counts'], 'self_test':coverage['self_test'],
            'full_coverage':coverage['all_membership_roles_identity_covered'],
            'group_linkage':coverage['review_group_id_linkage']},
        'support_rows_checked':len(support), 'minimum_remaining_train':min(int(r['retained_train_rows']) for r in support),
        'all_parent_intersections_zero':True,
        'pre_freeze_corrections':[
            'Initial uncertain-group projector required an immediately repeated axis prefix and missed host 24. The v1 script/output are retained but excluded from consumption; v2 uses the axis declared in the literal frozen group reference and covers seven members. No chemical rule or key changed.',
            'The mask constructor first stopped on a synthetic random-choice expected set {4,6}; the correct RandomState(100000) sample from sorted [1,2,4,6,9] is {1,9}. Only this synthetic fixture was corrected before any real-data construction. The real masks were constructed once.'],
        'boundary':{'rf_fits':0,'real_e2_calculated':False,'npz_arrays_loaded':False,
                    'real_predictions_parsed':False,'target_validity':'DEFERRED_TO_SEPARATE_STAGE_B'},
        'verification_limit':'Static implementation review, synthetic arithmetic, reused full constructor checks, and three sampled blocks. No claim of executed real E1/E2 equivalence.'})
    print(json.dumps({'status':'STAGE_A_IMPLEMENTATION_ACCEPTED','mask_sample_blocks':list(samples),
                      'minimum_remaining_train':min(int(r['retained_train_rows']) for r in support)}))


if __name__=='__main__':
    main()
