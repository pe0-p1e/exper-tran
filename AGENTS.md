# AGENTS.md — experiment/a100-precommercial-v6

This branch exists for one task: complete all open-source experiments before commercial-model evaluation.

Before changing code or launching jobs, read:

1. `experiments/2026-10-a100-precommercial-v6/AGENTS.md` — authoritative scientific and execution instructions.
2. `experiments/2026-10-a100-precommercial-v6/experiment_matrix.yaml` — exact machine-readable experiment matrix.
3. `experiments/2026-10-a100-precommercial-v6/README.md` — execution order and deliverables.

The nested `AGENTS.md` is authoritative for the entire branch, including edits under `src/`, `configs/`, and scripts outside the experiment directory.

Critical constraints:
- do not evaluate commercial models or call commercial APIs;
- do not redesign Multiple Proxy into Phase A/B/C;
- do not change 5 source-target pairs x 30 images;
- main Pull/Push weights are 0.75/0.25;
- main proxy representation uses the deepest visual block;
- TASR is target success conditioned on successful proxy targeted attack;
- target models are evaluation-only and must not influence attack generation;
- preserve V5 outputs; write all new artifacts under `outputs/a100_precommercial_v6/`.

Finish the full matrix and analysis, produce auditable summaries/reports, then stop before commercial-model work.


## V6 Multiple Proxy recovery override (2026-10-14)

The user clarified with the advisor that proxy intermediate image representations—not model output predictions—drive the attack. The original all-proxy clean-classification screening caused 17 missing multi-proxy transition cells and is **not an eligibility requirement** for this recovery. Before implementing anything, read:
- `experiments/2026-10-a100-precommercial-v6/MULTIPLE_PROXY_RECOVERY.md`
- `experiments/2026-10-a100-precommercial-v6/multiple_proxy_recovery.yaml`
- `experiments/2026-10-a100-precommercial-v6/V7_OPTIMIZATION_GATE.md`

For the **recovery campaign only**, these override older clean-valid screening and multi-proxy primary-metric instructions wherever they conflict:
- All proxy types (VLM, CLIP, SigLIP2, DINOv2) participate through their intermediate image embeddings/prototypes. Do not require proxy clean classification or agreement before attacking.
- Select a fixed per-source-class 30-image manifest from ground-truth ImageNet labels, disjoint from reference images; selection cannot depend on *any* model outputs, including the target.
- Keep frozen V6 results and original workbook unchanged. Execute the original two-block Multiple Proxy composition matrix on the same cohort for fair comparisons and save to a distinct output root.
- Report unconditional target targeted ASR, target-clean-correct-conditioned ASR, representation changes, and diagnostic legacy conditional TASR separately. Never rename ASR as TASR or count undefined denominators as zero.
- This does not relax the ban on target feedback/gradients during attack generation.

Do **not** preemptively create the optimization branch. First complete and assess the V6 recovery. Only if the documented quality gate triggers, create the separate V7 research branch to test additional proxies, gradient fusion, and input/noise optimization algorithms.
