# Figure 2: frozen annotation sources

This document maps the annotations in manuscript Figure 2 (panels A–C) to supplied frozen records. It does not reconstruct the complete historical figure generator. Panels A and B concern the 2,383-record structure-valid population under released-identifier splits. Panel C instead concerns distinct raw-SMILES entity appearances in the full 2,609-record release; its denominators must not be substituted for record or pair counts.

## Panel A: representative guest canonical-key recurrence and split summary

The [alias map](../data/identity/canonical_identity_alias_map_2026-08-01.tsv), selected by `entity_type=guest` and `raw_id=740` or `326`, records raw SMILES `C(C)[NH3+]` and `CC[NH3+]`, respectively. Both serialize to canonical SMILES `CC[NH3+]` and belong to alias group `GUEST_CAN_0380`.

The [released row inventory](../data/folds/folds_random.csv) binds `row_index=1198` to host 1 / guest 740 and `row_index=422` to host 8 / guest 326. For double-cold block 0, [held-out groups](../data/folds/folds_double_cold_heldout_groups.csv) contain neither host 1 nor guest 740, so row 1198 belongs to training. The [test membership](../data/folds/folds_double_cold_test.csv) and [recurrence audit](../data/identity/canonical_identity_recurrence_2026-08-01.tsv) identify row 422 as a recurrent test appearance in `seed_0`. The 2D chemical drawings illustrate guest identity across different host–guest records, not a 3D binding pose or an identical host–guest pair.

The split-level protected-axis recurrence summary reported with Panel A and the Figure 2 caption comes from the [canonical identity recurrence audit](../data/identity/canonical_identity_recurrence_2026-08-01.tsv), `record_type=method_summary`. The numerator is `protected_canonical_recurrence_appearances`; the denominator is `test_appearances`. Percentages round `100 × numerator / denominator` to three decimal places:

| Split | Recurrent / test appearances | Displayed percentage |
|---|---:|---:|
| host_cold | 2 / 2,383 | 0.084% |
| guest_cold | 25 / 2,383 | 1.049% |
| double_cold | 37 / 1,984 | 1.865% |

The protected axis is host for host-cold, guest for guest-cold, and either partner for double-cold. In double-cold evaluation, canonical-key recurrence occurs only on the guest axis (host recurrence is zero).

## Panel B: protected-axis recurrence across double-cold blocks

Source: the same [canonical identity recurrence audit](../data/identity/canonical_identity_recurrence_2026-08-01.tsv), `record_type=block_detail`, `split=double_cold`, where `block_id=seed_k` supplies block `k`, together with `record_type=method_summary` for the pooled `Total` row. Bar-end labels use `protected_canonical_recurrence_appearances / test_appearances`; host recurrence is zero in all 12 blocks. Because units are test appearances across overlapping blocks, a record appearing in multiple blocks can contribute more than once.

| Block | Figure label (`n / N`) |
|---:|---:|
| 0 | 7 / 100 |
| 1 | 0 / 128 |
| 2 | 4 / 244 |
| 3 | 1 / 165 |
| 4 | 2 / 170 |
| 5 | 0 / 197 |
| 6 | 0 / 159 |
| 7 | 1 / 76 |
| 8 | 9 / 232 |
| 9 | 2 / 107 |
| 10 | 9 / 195 |
| 11 | 2 / 211 |
| Total (pooled appearances) | 37 / 1,984 (1.865%; 9 / 12 blocks; 31 / 1,382 distinct test records) |

The `Total` row reports the pooled appearance ratio (`37 / 1,984`) rather than an unweighted mean of the 12 block percentages.

## Panel C: full-release nearest training analogs

Source: [saved full-release diagnostic](../results/diagnostics/suprabench_marginal_killgate_results.json), `similarity.<split>.<axis>.frac_gt_0.9`. The [retained producer](../scripts/diagnostics/suprabench_marginal_killgate.py), function `simdist`, operates on the full 2,609-record release, counts each distinct raw-SMILES entity once in each test fold separately for host and guest, and records the fraction with maximum training-set Tanimoto similarity strictly above 0.9. Host and guest percentages use their own fold-level entity denominators (unrecorded in the saved summary JSON) rather than record totals. The figure rounds these saved fractions to whole percentages:

| Split | Saved host fraction → display | Saved guest fraction → display |
|---|---:|---:|
| random | 0.931 → 93% | 0.697 → 70% |
| host_cold | 0.232 → 23% | 0.628 → 63% |
| guest_cold | 0.917 → 92% | 0.179 → 18% |

The saved similarity audit does not supply a double-cold analog-share estimate; double-cold is not recorded (and is not zero). In the 2 × 2 split icons for Panels B and C, crossed cells lie outside the panel diagnostic, and the four schemes represent independent split designs rather than subsets of a single partition. These close-analog shares are a separate descriptive quantity from the canonical-key recurrence in Panels A and B.
