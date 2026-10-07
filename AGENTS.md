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
