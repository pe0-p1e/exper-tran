# Budgeted V7 research progress

Budget: 20 hours from original start. Completed research conditions: 97. Failed attempts: 0. Remaining wall time: 15.78 h.

Development uses actual native outputs of held-out Qwen9B, InternVL8B and Gemma E4B. These are not final Qwen27B/InternVL14B/Gemma31B results. Initial pilot is ten GT-selected images, two per direction; larger development confirmation is explicitly marked. No prediction-based screening.

| Condition | Worst model-direction ASR | Pooled native ASR | Images/direction |
|---|---|---|---|
| P00 | 50.0% | 83.3% | 2 |
| P01 | 50.0% | 80.0% | 2 |
| P02 | 0.0% | 83.3% | 2 |
| P03 | 50.0% | 80.0% | 2 |
| T10_L00 | 0.0% | 40.0% | 2 |
| T10_L01 | 0.0% | 43.3% | 2 |
| T10_L02 | 0.0% | 46.7% | 2 |
| T10_L03 | 50.0% | 90.0% | 2 |
| T10_L04 | 0.0% | 50.0% | 2 |
| T10_L05 | 50.0% | 86.7% | 2 |
| T10_L06 | 50.0% | 80.0% | 2 |
| T10_L07 | 50.0% | 83.3% | 2 |
| T10_M00 | 50.0% | 90.0% | 2 |
| T10_M01 | 50.0% | 90.0% | 2 |
| T10_M02 | 50.0% | 86.7% | 2 |
| T10_M03 | 50.0% | 90.0% | 2 |
| T10_M04 | 50.0% | 96.7% | 2 |
| T10_M05 | 50.0% | 86.7% | 2 |
| T10_M06 | 0.0% | 60.0% | 2 |
| T10_C00 | 50.0% | 96.7% | 2 |
| T10_C01 | 50.0% | 93.3% | 2 |
| T10_C02 | 50.0% | 93.3% | 2 |
| T10_C03 | 50.0% | 83.3% | 2 |
| T10_C04 | 50.0% | 93.3% | 2 |
| T10_C05 | 50.0% | 96.7% | 2 |
| T10_C06 | 0.0% | 73.3% | 2 |
| T10_C07 | 50.0% | 90.0% | 2 |
| T10_single_2eca981e | 0.0% | 60.0% | 2 |
| T10_single_d0d3bf33 | 0.0% | 63.3% | 2 |
| T10_O00 | 50.0% | 90.0% | 2 |
| T10_O01 | 50.0% | 93.3% | 2 |
| T10_O02 | 50.0% | 90.0% | 2 |
| T10_O03 | 50.0% | 90.0% | 2 |
| T10_O04 | 0.0% | 80.0% | 2 |
| T10_O05 | 0.0% | 63.3% | 2 |
| T10_O06 | 50.0% | 90.0% | 2 |
| T10_O_noDI | 0.0% | 40.0% | 2 |
| T10_O_TI | 0.0% | 73.3% | 2 |
| T20_layer_single_0 | 0.0% | 0.0% | 2 |
| T20_layer_single_1 | 0.0% | 56.7% | 2 |
| T20_layer_single_2 | 50.0% | 83.3% | 2 |
| T20_layer_single_3 | 0.0% | 43.3% | 2 |
| T20_layer_without_0 | 50.0% | 90.0% | 2 |
| T20_layer_without_1 | 0.0% | 86.7% | 2 |
| T20_layer_without_2 | 0.0% | 66.7% | 2 |
| T20_layer_without_3 | 50.0% | 90.0% | 2 |
| T20_layer_late_pair | 0.0% | 83.3% | 2 |
| T20_layer_middle_pair | 50.0% | 83.3% | 2 |
| T20_layer_spread_pair | 0.0% | 33.3% | 2 |
| T20_layer_early_weight | 50.0% | 83.3% | 2 |
| T20_layer_late_weight | 50.0% | 90.0% | 2 |
| T20_layer_middle_weight | 50.0% | 83.3% | 2 |
| T20_reference_0 | 50.0% | 83.3% | 2 |
| T20_reference_1 | 50.0% | 93.3% | 2 |
| T20_reference_2 | 50.0% | 96.7% | 2 |
| T20_reference_3 | 50.0% | 86.7% | 2 |
| T20_interaction_0 | 0.0% | 36.7% | 2 |
| T20_interaction_1 | 0.0% | 30.0% | 2 |
| T20_interaction_2 | 50.0% | 93.3% | 2 |
| T20_interaction_3 | 50.0% | 90.0% | 2 |
| T20_interaction_4 | 0.0% | 76.7% | 2 |
| T20_interaction_5 | 0.0% | 76.7% | 2 |
| T10_R00 | 50.0% | 93.3% | 2 |
| T10_R01 | 50.0% | 90.0% | 2 |
| T10_R02 | 50.0% | 90.0% | 2 |
| T10_R03 | 50.0% | 93.3% | 2 |
| T10_R04 | 50.0% | 90.0% | 2 |
| T10_R05 | 50.0% | 90.0% | 2 |
| T10_R06 | 50.0% | 93.3% | 2 |
| T10_R07 | 50.0% | 93.3% | 2 |
| T10_R08 | 50.0% | 90.0% | 2 |
| T10_R09 | 50.0% | 93.3% | 2 |
| T10_R10 | 50.0% | 90.0% | 2 |
| T10_R11 | 50.0% | 86.7% | 2 |
| T10_R12 | 50.0% | 90.0% | 2 |
| T10_R13 | 50.0% | 93.3% | 2 |
| T10_R14 | 50.0% | 90.0% | 2 |
| T10_R15 | 50.0% | 93.3% | 2 |
| T10_R16 | 50.0% | 90.0% | 2 |
| T10_R17 | 50.0% | 93.3% | 2 |
| T10_R18 | 50.0% | 90.0% | 2 |
| T10_R19 | 50.0% | 90.0% | 2 |
| T10_R20 | 50.0% | 90.0% | 2 |
| T10_R21 | 50.0% | 93.3% | 2 |
| T10_R22 | 0.0% | 86.7% | 2 |
| T10_R23 | 50.0% | 93.3% | 2 |
| T20_component_0 | 0.0% | 60.0% | 2 |
| T20_component_1 | 0.0% | 70.0% | 2 |
| T20_complementarity_anchor | 50.0% | 96.7% | 2 |
| T10_confirm_0 | 75.0% | 96.7% | 8 |
| T10_confirm_1 | 62.5% | 91.7% | 8 |
| T10_confirm_2 | 75.0% | 94.2% | 8 |
| T10_seed_43 | 62.5% | 95.0% | 8 |
| T10_seed_44 | 87.5% | 96.7% | 8 |
| T20_matched_single_0 | 12.5% | 60.8% | 8 |
| T20_matched_single_1 | 12.5% | 65.8% | 8 |
| T20_matched_single_2 | 0.0% | 70.8% | 8 |

A high two-image pilot is not evidence for the formal >=15/30 criterion. Undefined conditional TASR is N/A. All added EOT/variance/restart computation must be interpreted with compute_budget.csv. Failures and budget-skipped conditions are retained.

## Frozen final target evaluation

| Condition | Model | Direction | Hits/N |
|---|---|---|---|
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T01 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T03 | 28/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T05 | 25/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T07 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T09 | 25/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 29/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 28/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T01 | 29/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T03 | 27/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T05 | 23/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T07 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T09 | 21/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T01 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T03 | 26/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T05 | 20/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T07 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T09 | 13/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 29/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 25/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 18/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 11/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T01 | 15/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T03 | 16/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T05 | 8/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T07 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T09 | 5/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T01 | 14/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T03 | 1/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T05 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T07 | 4/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T09 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 26/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 1/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 3/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 13/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 2/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T01 | 1/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T03 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T05 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T07 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T09 | 0/30 |

Missing/partial conditions are not zeros. Nested dev confirmation overlaps the initial pilot and is not an untouched generalization set. The final test cohort is independent and unchanged.

## Final execution validation

Process exit: 0; research conditions: 97; failed attempts: 0. All final cells complete: True. Selected method meets >=15/30 in all 15 cells: True.

| Condition | Target | Direction | Hits/N | Clean-correct hits/N |
|---|---|---|---|---|
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T01 | 30/30 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T03 | 28/30 | 28/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T05 | 25/30 | 25/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T07 | 30/30 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Qwen3.5-27B | T09 | 25/30 | 25/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 30/30 | 28/28 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 29/30 | 28/29 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 30/30 | 27/27 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 30/30 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 28/30 | 26/28 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T01 | 29/30 | 29/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T03 | 27/30 | 27/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T05 | 23/30 | 23/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T07 | 30/30 | 30/30 |
| focus_T10_final_0__A3__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__DI__s42__i50__r1 | Gemma 4 31B-it | T09 | 21/30 | 21/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T01 | 30/30 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T03 | 26/30 | 26/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T05 | 20/30 | 20/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T07 | 30/30 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Qwen3.5-27B | T09 | 13/30 | 13/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 29/30 | 27/28 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 25/30 | 25/29 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 18/30 | 17/27 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 30/30 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 11/30 | 11/28 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T01 | 15/30 | 15/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T03 | 16/30 | 16/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T05 | 8/30 | 8/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T07 | 30/30 | 30/30 |
| focus_T10_final_1__A3__equal__Qwen3.5-2B__DI__s42__i50__r1 | Gemma 4 31B-it | T09 | 5/30 | 5/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T01 | 14/30 | 14/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T03 | 1/30 | 1/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T05 | 0/30 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T07 | 4/30 | 4/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Qwen3.5-27B | T09 | 0/30 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T01 | 26/30 | 24/28 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T03 | 1/30 | 1/29 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T05 | 3/30 | 3/27 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T07 | 13/30 | 13/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | InternVL3.5-14B-HF | T09 | 2/30 | 2/28 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T01 | 1/30 | 1/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T03 | 0/30 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T05 | 0/30 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T07 | 0/30 | 0/30 |
| focus_T10_final_pullpush__A0__equal__CLIP_ViT-L_14+Qwen3.5-2B+InternVL3.5-2B-HF__s42__i50__r1 | Gemma 4 31B-it | T09 | 0/30 | 0/30 |

Legacy conditional TASR is N/A because no all-proxy native-output hit event was measured. Missing measurements are not zero. Nested development confirmation is not an untouched generalization set. The final test cohort is independent and unchanged. Actual wall runtime is in end_of_budget.json; the 20-hour budget is an upper limit, not measured GPU compute. Gradient diagnostics describe pairwise cosine negativity, not pixelwise sign disagreement; cancellation ratio is measured for the unfused equal mean even when a different fusion is used. Trace endpoints are before their respective update.
