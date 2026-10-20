# Direction differences persist while the broader composition campaign remains incomplete

Checkpoint collected at 20261020T074117Z UTC. This is a live partial-campaign snapshot; it does not mark pending work as finished.

## Coverage

Composition conditions: {"ATTACK_COMPLETE": 6, "COMPLETE": 60, "PARTIAL": 3, "QUEUED": 27}. Total declared rows: 96.
Follow-up controlled conditions: 5/11 complete.
Full per-condition, per-direction and paired comparisons: [composition reports](../proxy_study_18h/reports/).
Detailed frozen target batches, manifests, settings and BF16 audits: [snapshot data](data/).
Attack batch identities, layer settings, counters and PNG hashes: [metadata](attack_batch_metadata.jsonl).

## Commercial evaluation

| Requested model | Returned adversarial hits / observed images | Full cohort status |
|---|---:|---|
| gpt-6.1-sol | 90/150 | COMPLETE |
| gpt-6-sol | 117/150 | COMPLETE |
| gpt-5.6-sol | 0/0 | PARTIAL / no full ASR |
| gpt-6-astra | 28/42 | PARTIAL / no full ASR |
| gpt-6-luna | 0/0 | PARTIAL / no full ASR |
| gpt-5.6-terra | 0/0 | PARTIAL / no full ASR |
| gpt-5.6-luna | 0/0 | PARTIAL / no full ASR |
| gemini-3.8-flash-low | 104/150 | COMPLETE |
| gemini-3.8-flash-high | 102/150 | COMPLETE |
| gemini-3.8-flash-medium | 102/150 | COMPLETE |
| claude-sonnet-4-6 | 38/49 | PARTIAL / no full ASR |
| claude-opus-4-6-thinking | 0/0 | PARTIAL / no full ASR |

Returned-image rates on incomplete cohorts are descriptive partial observations. They must not be treated as full-cohort ASR. Gemini entries are reasoning profiles of one generation. CLI IDs do not establish undisclosed backend snapshots.

## Scientific settings and interpretation

The composition study uses A3: four visual depths (25/50/75/100%), ten-class prototype margin, local patch weight 0.35, DI, and equal normalized proxy gradients. Fixed GT-selected 5×30 images, 48 disjoint reference images/class, seed42, 50 steps, L∞16/255, step1/255 and momentum1. All participating open-weight model parameters are BF16, without quantization. Targets evaluate frozen PNGs only.
The follow-up loss/fusion controls use a separate ten-image surrogate development cohort. Their results are not final-target test scores. Commercial feedback does not select loss, proxies or attack parameters.
Direction differences are the main descriptive finding. Reverse-direction measurements are needed before claiming directional asymmetry. Reference CKA/RSA, class dispersion, gradient conflicts, ΔR and margins are associations rather than causal explanations. PCA/t-SNE groups use a shared fit; t-SNE is qualitative.

## Measurement artifacts

[Reference geometry](../unified_campaign/mechanism_reference/), [gradient compatibility](../unified_campaign/mechanism_gradients/), [representation measurements and joint coordinates](../unified_campaign/surrogate_followup/analysis/), [commercial per-request provenance](../unified_campaign/commercial_request_provenance.json).

## Storage and remaining work

Full local artifacts remain at `/home/ubuntu/exper-tran/outputs/a100_precommercial_v7_targeted50` on `/dev/vda1 /`. The storage audit found zero output links into `/ephemeral`. PNGs, NPZ feature arrays, complete optimization traces, raw CLI traces and resumable states remain local. OAuth credentials are outside this publication.
Local inventory: `/home/ubuntu/exper-tran/outputs/a100_precommercial_v7_targeted50/proxy_composition_18h/snapshots/20261020T074117Z/local_result_inventory.csv`. Model weight caches may remain on ephemeral storage. Root persistence covers ordinary hibernation/restart while the persistent disk is retained; VM/disk deletion is not an independent backup.
Running jobs are not paused by this checkpoint. Pending composition cells, incomplete follow-up controls, seeds and blocked commercial calls retain their explicit status. No new inference was launched by the snapshot exporter.
