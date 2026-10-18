# Twenty-hour extended V7 campaign

User approved additional research and a total 20-hour budget on 2026-10-18.
The default `run_all_v7.sh` and explicit `run_twenty_hour_v7.sh` continue the same campaign.
The `ten_hour_research/` output path and `T10_*` identifiers are retained for resume compatibility.
The original start is preserved; this extends the total reservation to 20 hours, not another 20 hours on every restart.
The finite queue may finish early. Six hours are reserved for confirmation, final attacks/evaluation and geometry; this is a scheduling allowance, not a completion guarantee.

## Additional controlled research

- Four single depths, four leave-one-layer-out conditions, three two-layer conditions and three layer-weight profiles on the same original three proxies: 14 conditions.
- Four soft-reference temperatures with all 48 reference images retained, on a frozen selected anchor.
- Equal versus PCGrad fusion crossed with no transform, DI, and DI+TI: six conditions on a fixed anchor.
- Every selected proxy individually, every leave-one-proxy-out subset and a matching full ensemble anchor.
- Top three ensembles confirmed on eight images/direction. The fixed nested development subset overlaps the pilot; do not call it untouched validation.
- Winning ensemble checked with seeds 43/44 on eight images/direction. These checks do not select a favorable seed; the official primary seed stays 42.
- Every member of the final winning ensemble gets a matched eight-image/direction single-proxy confirmation. The strongest validated single becomes the full-cohort baseline.

Rank on the worst surrogate model-direction ASR followed by pooled ASR. Keep 50 steps, epsilon16/255, BF16 and fixed ground-truth cohorts. Additional adaptive/EOT forwards are reported separately. No target feedback can alter selection.

## Measurements and reporting

The finalizer exports `gradient_mechanism_diagnostics.csv` with pairwise gradient cosine, negative-cosine fraction, unfused equal-mean cancellation, per-proxy initial/last-preupdate losses and margins, runtime and peak memory. Negative cosine is not pixelwise sign disagreement. Different losses have different numerical scales; raw losses must not be compared as a shared quality score.

Model-specific margins, CKA/RSA, representation shift and joint PCA/t-SNE are post-hoc and never used to tune against final targets. Budget-skipped computations remain explicitly incomplete. Final original 30-image/direction evaluation compares selected ensemble, strongest validated single and original Pull/Push, on Qwen27B, InternVL14B and Gemma31B.

All completed scientific artifacts and checkpoints stay on persistent root storage. Git uploads contain compact results/reports/audits and a few real examples, not all PNGs or model caches. Deferred COCO/Web studies remain outside this campaign.
