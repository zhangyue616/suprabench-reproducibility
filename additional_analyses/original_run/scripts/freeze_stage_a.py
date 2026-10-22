"""Seal Stage A, binding future inputs opaquely after fixing all mask identities."""
from __future__ import annotations
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from staged_io import file_identity, write_json

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]  # public repository root
PUBLIC = PAPER
COMMIT = '81b1d092b4d2bdd4a4f293f340e7d369395ac053'
ARCHIVE = PAPER/f'suprabench-cold-split-audit_{COMMIT}.zip'  # historical input archive; not bundled
PREFIX = f'suprabench-cold-split-audit-{COMMIT}/'
FUTURE_INPUTS = {
    'canonical_features':'data/canonical_cpu/features/tabular_features.npz',
    'canonical_baseline_predictions':'results/canonical_cpu/run_a/predictions_cpu.csv',
    'canonical_baseline_metrics':'results/canonical_cpu/run_a/metrics_cpu.tsv',
    'family_membership':'data/folds/folds_family_cold.csv',
    'family_predictions':'data/released_predictions/predictions_family_cold.csv',
}


def opaque_archive_identity(archive, member):
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return {'bytes':archive.getinfo(member).file_size,'sha256':digest.hexdigest()}


def main():
    for name in ('analysis_contract.json','stage_a_freeze.json','README.md'):
        if (ROOT/name).exists():
            raise FileExistsError('refusing to replace '+name)
    acceptance = json.loads((ROOT/'acceptance/stage_a_acceptance.json').read_text(encoding='utf-8'))
    assert acceptance['status']=='STAGE_A_IMPLEMENTATION_ACCEPTED'
    projection = json.loads((ROOT/'inputs/projection_receipt.json').read_text(encoding='utf-8'))
    assert Path(projection['snapshot_zip'])==ARCHIVE
    recipes = json.loads((ROOT/'inputs/frozen_recipes.json').read_text(encoding='utf-8'))
    coverage = json.loads((ROOT/'masks/identity_coverage.json').read_text(encoding='utf-8'))
    with (ROOT/'masks/block_arm_support.csv').open(encoding='utf-8',newline='') as handle:
        support = list(csv.DictReader(handle))
    parent = [r for r in support if r['arm']=='parent_purge']
    # This is the mask freeze line. No outcome-containing source is opened above.
    frozen = {}
    for path in sorted((ROOT/'masks').glob('*')):
        frozen[path.relative_to(ROOT).as_posix()] = {'kind':'mask',**file_identity(path)}
    mask_fixed_utc = datetime.now(timezone.utc).isoformat()
    sources = {}
    with ZipFile(ARCHIVE) as archive:
        for source_id, relative in FUTURE_INPUTS.items():
            path = PUBLIC/relative
            local = file_identity(path)
            member = PREFIX+relative
            archived = opaque_archive_identity(archive, member)
            if local!=archived:
                raise RuntimeError('local source differs from the fixed snapshot archive: '+relative)
            sources[source_id] = {'path':path.as_posix(),**local,'archive_member':member,
                                  'binding_method':'opaque streaming digest; content not parsed in Stage A'}
    for source_id, relative in (('e1_train_masks','masks/train_masks.csv'),
                               ('e1_block_map','masks/block_map.csv'),
                               ('e1_block_arm_support','masks/block_arm_support.csv')):
        sources[source_id] = {'path':(ROOT/relative).as_posix(),
                              **{k:frozen[relative][k] for k in ('bytes','sha256')}}
    arms = ['baseline','parent_purge','random_purge_0','random_purge_1','random_purge_2']
    nulls = ['sim_knn_k5','additive','host_only_RF','guest_only_RF','cond_only_RF','source_only_RF']
    core_sources = {}
    for name in ('run_external_fold_models.py','common.py'):
        path = PAPER/'historical-source-not-redistributed'/name
        core_sources[name] = {'path':path.as_posix(),**file_identity(path),
                             'role':'static source reference; never imported by the new runner'}
    contract = {
        'schema_version':'e1e2-analysis-contract-v1',
        'experiment_root':ROOT.as_posix(),
        'stage':'A_FROZEN_BEFORE_STAGE_B',
        'decision_path':'not_redistributed:scientific-decision-record',
        'scientific_snapshot':{'commit':COMMIT,'zip':ARCHIVE.as_posix(),
                               'verification_scope':'only actual source members; no whole-archive rehash'},
        'sources':sources,
        'identity_sources_receipt':'inputs/projection_receipt.json',
        'canonical_formula_sources':core_sources,
        'execution':{'interpreter':'python',
            'reused_environment_preflight':{'python':'3.11.15','numpy':'1.26.4','pandas':'3.0.3',
                                          'scipy':'1.17.1','sklearn':'1.9.0','rdkit':'2026.03.3'},
            'order':['E2','E1'],'device':'CPU',
            'thread_environment':{k:'1' for k in ('BLIS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'results_policy':'new absolute child directory of experiment_root/results; reject existing path',
            'restart_policy':'no automatic resume/refit; retain failed evidence',
            'preflight_fit':False,'separate_stage_b_run_required':True},
        'e1':{
            'identity_layer':'v2_reviewed_only_parent','uncertain_groups':'three frozen groups; seven members retain member-specific strict keys',
            'uncertain_link_authority':'inputs/identity/uncertain_group_links_v2.csv',
            'canonical_to_parent_global_uniqueness_required':False,
            'row_index':'zero-based physical clean_cond row, never parquet_row',
            'n_blocks':12,'n_original_rows_per_block':2383,'test_appearances_per_arm':1716,
            'block_order':'numeric (seed,fold_index), b=0..11; canonical block_id=double_cold/seed_{fold_index}; fold_index=seed',
            'membership':'fixed original train/test/neither from canonical membership; no rebuilding or test exclusion',
            'arms':arms,'models':['PAIR_RF',*nulls],'nulls':nulls,
            'parent_deletion':'delete original train if its host parent is in test host parents OR guest parent is in test guest parents',
            'random_deletion':{'rng':'numpy.random.RandomState','seed_formula':'100000+100*b+j',
                'j':[0,1,2],'population':'sorted original train row_index',
                'size':'exact number removed by same-block parent purge','replace':False,
                'training_order':'sorted retained row_index'},
            'minimum_train_rows':50,'minimum_test_rows':25,'minimum_distinct_finite_test_targets':2,
            'target_gate':'check at Stage B, after frozen masks; no drop or replacement of invalid blocks',
            'rf_recipe':recipes['rf'],'maximum_rf_fits':300,'actual_planned_rf_fits':300,
            'rf_models':['PAIR_RF','host_only_RF','guest_only_RF','cond_only_RF','source_only_RF'],
            'additive':recipes['additive'],'sim_knn_k5':recipes['sim_knn_k5'],
            'input_precision':'retain original NPZ array dtypes and original scoring target dtype; no implicit casts',
            'all_nulls_refit_or_recomputed':'own arm training rows, own additive mean and own k5 library; never use saved null scores',
            'guarded_r2':'target has at least two distinct values and prediction std > 1e-9; finite r2 required',
            'best_null':'maximum finite eligible guarded R2 among exactly the same six nulls; no eligible null means undefined',
            'standard_auxiliary':'standard R2 including constant predictions, same six nulls; no extra mean-prediction model',
            'margin':'PAIR_RF R2 minus same-arm best-null R2',
            'aggregate':'all 12 blocks equal weight; any undefined block makes the corresponding arm mean undefined',
            'primary_contrast':'parent-purge equal-block margin mean minus mean of three random-purge equal-block margin means',
            'secondary_contrast':'parent-purge equal-block margin mean minus baseline equal-block margin mean',
            'baseline_check':'all original baseline blocks before intervention fits; exact keys and saved target strings, finite numeric differences <= 8*ulp(reference); unexplained mismatch stops E1',
            'report':'all blocks/arms, absolute RF and six-null metrics, MAE, guarded/standard margins, deletion share and remaining exposure; first formal block time and process peak memory',
            'inference':'descriptive; no p value/CI; overlapping blocks and random masks are not independent chemical replicates'},
        'e2':{
            'models':['PAIR_RF','guest_only_RF'],'expected_records':352,'expected_hosts':21,
            'membership_selection':{'split':'family_cold','held_family':'cyclodextrin','role':'test'},
            'panel':'all membership-selected records; require one prediction per model per row, aligned raw host/guest and target',
            'target_matching':'exact float64 equality or exact source float32 round trip, no broad tolerance',
            'target_source':'family_membership.y',
            'rank_definition':'average-tie global ranks for each target and prediction vector on all 352 records',
            'denominator':'sqrt(sum((global_y_rank-mean)^2)*sum((global_pred_rank-mean)^2))',
            'between':'sum_h n_h*(mean_global_y_rank_h-mean_global_y_rank)*(mean_global_pred_rank_h-mean_global_pred_rank)/denominator',
            'within':'sum_h sum_i_in_h (global_y_rank_i-mean_global_y_rank_h)*(global_pred_rank_i-mean_global_pred_rank_h)/denominator',
            'check':'between+within equals pooled Spearman and separately scipy.stats.spearmanr',
            'primary_contrast':'PAIR_RF signed within contribution minus guest_only_RF signed within contribution',
            'host_min_distinct_guests':5,
            'secondary_eligibility':'at least five distinct raw guests and both host-local Spearman values defined',
            'secondary_weight':'eligible raw hosts equally weighted, paired model rho difference',
            'support_table':'all 21 hosts including ineligible, distinct guest counts, ties, constant predictions/targets and undefined reasons',
            'no_fit':True,'no_inference':'no p values/CI; test-host centering is a decomposition, not calibration'},
        'prohibitions':['no GPU/GNN','no hyperparameter search','no rule/seed adjustment from outcomes',
                        'no MSE decomposition, pair averaging or 17-model scan',
                        'no edits to the fixed snapshot or historical files',
                        'no dependency installation'],
        'no_study_wide_blinding_claim':True,
        'post_freeze_bug_policy':'retain prior code and reason, targeted repair only; frozen masks/rules may not be silently changed'}
    write_json(ROOT/'analysis_contract.json', contract)
    table = ['| b / seed / fold | original train | test | neither | parent deleted | retained train | random seeds |',
             '|---:|---:|---:|---:|---:|---:|:---|']
    for row in parent:
        b = int(row['b'])
        table.append('| '+ ' | '.join([str(b), row['original_train_rows'],row['original_test_rows'],
            row['original_neither_rows'],row['deleted_train_rows'],row['retained_train_rows'],
            ', '.join(str(100000+100*b+j) for j in range(3))])+' |')
    readme = ('# Additional analyses: Stage A freeze\n\n'
        'Status: `STAGE_A_FROZEN_READY_FOR_SEPARATE_STAGE_B`. Implementation checks passed for code, identities and masks only.\n\n'
        'Contract: `analysis_contract.json`; frozen identities: `stage_a_freeze.json`; consolidated checks: `acceptance/stage_a_acceptance.json`. '
        'E2 and E1 main scripts: `scripts/e2_rank_diagnostic.py` and `scripts/e1_parent_purge_runner.py`. '
        'Full five-arm support table: `masks/block_arm_support.csv`; row-level masks: `masks/train_masks.csv`.\n\n'
        'All 189 raw hosts / 1,164 raw guests are covered, with no raw-ID canonical SHA or primary-parent conflict. '
        'The seven members of the three uncertain groups keep their own strict keys; no global one-to-one canonical-key-to-parent mapping is imposed. '
        'Each block keeps the original roles of all 2,383 rows; the 12 blocks contain 1,716 test appearances. Both parent-arm axis intersections are zero; '
        'each random arm deletes as many rows per block as the parent arm; the minimum remaining train size over 60 block-arms is 900, above the original 50-row minimum.\n\n'
        +'\n'.join(table)+'\n\n'
        'Constructor full-coverage checks were reused; eight synthetic E2 cases and the no-fit E1 self-test were run, and the masks of blocks 0, 4 and 11 were independently spot-checked. '
        'The v1 group-link projection missed host 24; the v1 script and output are retained and the seven-member v2 links are used; '
        'a wrong expected value in a synthetic random-number fixture halted the run before real construction; only the fixture was corrected. Real masks were built once.\n\n'
        'Mask SHA-256 values were fixed first; the five Stage B inputs were then bound by ZIP/local byte digests without parsing content. No NPZ arrays were loaded, '
        'no real predictions were parsed, no model was trained and E2 was not computed; target finiteness/variability and real-baseline agreement are deferred to Stage B. '
        'Mask construction read only identities and membership; earlier results were already known, so no whole-study blinding is claimed.\n\n'
        'Stage B runs E2 and then E1 under the frozen contract. The first real block records its run time; '
        'all fits count toward the 300-fit cap; a failure is retained and stops the affected E1 run, without overwriting or automatic retry. '
        'Completing Stage A is not a scientific conclusion.\n')
    with (ROOT/'README.md').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(readme)
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in frozen or relative=='stage_a_freeze.json':
            continue
        kind = 'script' if relative.startswith('scripts/') else ('contract' if relative=='analysis_contract.json' else 'support')
        frozen[relative] = {'kind':kind,**file_identity(path)}
    receipt = {
        'status':'STAGE_A_FROZEN_READY_FOR_SEPARATE_STAGE_B',
        'completed_utc':datetime.now(timezone.utc).isoformat(), 'mask_fixed_before_opaque_outcome_binding_utc':mask_fixed_utc,
        'acceptance':'acceptance/stage_a_acceptance.json','contract':'analysis_contract.json',
        'frozen_files':frozen, 'source_bindings':sources,
        'identity_counts':coverage['identity_counts'], 'uncertain_group_members':7,'uncertain_groups':3,
        'original_test_appearances':1716,'block_count':12,'arm_support_rows':60,'mask_rows':142980,
        'parent_support':parent,
        'all_parent_same_axis_intersections_zero':True,'all_arms_minimum_train_pass':True,
        'minimum_remaining_train':900,
        'boundary':{'rf_fits':0,'new_e2_results':False,'npz_arrays_loaded':False,'real_predictions_parsed':False,
                    'opaque_source_bytes_hashed_after_mask_fix':True,'stage_b_started':False},
        'stage_b_plan':'separate Stage B: E2 then E1, frozen masks and rules unchanged'}
    write_json(ROOT/'stage_a_freeze.json',receipt)
    print(json.dumps({'status':receipt['status'], 'frozen_files':len(frozen),
                      'contract_identity':frozen['analysis_contract.json'],
                      'mask_identity':frozen['masks/train_masks.csv']}))


if __name__=='__main__':
    main()
