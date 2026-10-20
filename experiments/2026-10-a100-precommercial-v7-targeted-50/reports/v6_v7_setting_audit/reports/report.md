# V6→V7 setting and implementation audit

## Correct baseline: matched Multiple Proxy, not the single-proxy layer table

The primary comparator is immutable BF16 V6 Multiple Proxy recovery versus the
same original E01–E13 rows in current V7. Both use GT-only fixed150-image cohorts,
48 references/class, pinned weights and unconditional target ASR. Original V6
single-proxy layer selection and proxy-conditioned TASR are separate evidence.

| Matched original row | V6 recovery hits/150 | Current four-layer PP hits/150 |
|---|---:|---:|
| E03 Intern2B+4B+8B→Qwen27B | 5 |132 |
| E09 GemmaE2B+E4B+26B→InternVL14B |21 |92 |
| E11 Intern2B+GemmaE2B+CLIP→Qwen27B |10 |129 |
| E13 E11+DINO→Qwen27B |8 |135 |

V6 recovery already had multiple proxies, equal per-image L1-normalized fusion,
BF16 and the cosine prototype Pull/Push objective. None is newly introduced by
V7. Its multi-proxy runner fixes each proxy to its deepest block. The existence
of a different single-proxy layer-sweep table does not establish an ensemble
layer-selection experiment.

## Inventory and independently checked identities

`setting_inventory.csv` lists50 known setting/execution categories across
original V6, recovery V6, historical V7 A3 and current V7 pure PP. A list of
possible causes is not a completed causal attribution.

* Same five test manifests,150 test IDs,ten reference manifests and480 reference
  IDs.630 canonical image files are byte-identical between recovery and V7.
* All15 original model specifications/revisions match. Recovery has198 saved
  parameter audit events, all pass, floating parameters BF16.
* All1,950 matched clean target responses and parsed labels agree between the
  recovery and current formal original13 evaluations.24 missing parsed labels
  occur in BOTH versions; CSV empty and JSON null are equivalent, not a change
  or a zero. Strict parser handling is retained.
* Target prompt/parser paths and formal clean/adv batch2 behavior are shared.
  Development cached-clean evaluation may use adv batch1 and is disclosed
  separately. A clean-response match does not prove every adversarial response
  is invariant to execution batch.

## Actual feature/gradient and historical-cache checks

Same loaded BF16 weights, same two development images,75% and deepest actual
blocks of Qwen2B,Intern2B,GemmaE2B:

* All six embedding comparisons have maximum absolute difference0.
* Gradient cosine minima across compared images exceed0.9999. Small backward
  differences remain; this is numerical agreement, not bitwise full-attack
  identity. Both compared losses use .75/.25 and a shared reference bank.
* Three historical deepest-layer banks,480 vectors/model, were compared to the
  current cached banks. Qwen/Intern have small numeric differences; Gemma is
  identical. Source-target prototype-difference direction cosine is at least
  .99957 for Qwen, .999997 for Intern, and essentially1 for Gemma.

This is current-checkout V6 API/V7 path parity plus actual old saved feature
values, NOT a historical binary/environment replay. The old complete software
stack is not reproduced. A subtle difference can still accumulate over50 sign
updates; cache-substitution attack and historical-environment replay remain
unperformed. These local checks reduce suspicion of a gross pooling/tap change
for these three models but do not rule out every model or numerical effect.

## Completed small ablations: same dev10 and checkpoint-held-out validators

A=CLIP+Qwen2B+Intern2B; B=CLIP+Qwen2B+GemmaE2B. Each count below is30 correlated
model-image observations: ten images times three validators. Some validators
share a family with an attack proxy. These are NOT the exact original E11 formal
target measurements, and cannot be subtracted from that V6 table.

| Setting | A hits/30 | B hits/30 |
|---|---:|---:|
| Four layers,DI,.5/.5 reference |29 |28 |
| DI off |16 |21 |
| Raw gradient mean,normalization removed |28 |24 |
|75% single layer per proxy,still THREE proxies |26 |23 |
|50/75% pair |29 |18 |
|50/75/100% triple |28 |29 |

The same two-layer rule does not work equally across compositions. For raw
fusion,A mean raw L1 shares are40.1/30.5/29.4%;B shares are56.2/40.9/3.0% in
model order CLIP,Qwen,Gemma. Raw-versus-normalized fused direction cosine averages
.962 for A and .847 for B along each raw-control trajectory. This supports a
scale-dominance hypothesis for B, not proof of causal mediation. CurrentV6
recovery already normalizes, so normalization cannot explain its jump to V7.

## Remaining causal bridge, now running separately

`run_v6_v7_multi_bridge.sh` fixes the exact original E11 order
Intern2B+GemmaE2B+CLIP. It uses the same ten development images and Qwen9 as a
whole-family-excluded held-out surrogate, NEVER final Qwen27 feedback.
Seven prespecified50-step arms:

1. deepest,noDI,.75/.25 (V6-like settings in current validated runner);
2.75%,noDI,.75/.25;
3. deepest,DI,.75/.25;
4.75%,DI,.75/.25;
5. four layers,noDI,.75/.25;
6. four layers,DI,.75/.25;
7. four layers,DI,.5/.5.

No proxy is added or removed across these rows.75% is a representative tap, not a
final-target-selected optimum. This isolates layer identity/aggregation,DI and
coefficient changes on the matched MULTI composition. It does not replay the
historical software binary and does not by itself explain all13 ensembles.
No full-cohort expansion follows automatically. All unresolved causes remain
labeled unresolved; interactions must not be assigned a single cause without
appropriate contrasts.
