# V8 Stage 1–2 handoff

Status: RUNNING. Completed evaluations: 203/236 planned so far.

Algorithm: strict BF16, native spatial-token mean features,48 references/class, .5 cosine target pull + .5 source push,50 steps,L∞16/255,DI, equal per-image L1-normalized proxy gradients. Layer losses are averaged inside each model before proxy normalization. No output-head, patch or margin loss.

Family exclusion applies to each actual validator, including Qwen9/InternVL8/GemmaE4. Final open-weight target150-image cohorts remain sealed. Historical10-image screening is development data; exploratory30 and confirmation50 are separately frozen. Stage2 reuses confirmation images only after selection lock.

BF16 audit load events: 753. Failed audit events: 0.

Failures retained: 4. See scheduler.json and attempts/*/exit.json; missing results are not zeros.

## Frozen methods
{
  "protocol": "V8_integrated_ILA_FDA_geometry_joint_selection",
  "families": {
    "qwen": {
      "primary": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "InternVL3.5-2B-HF": [
          5,
          11,
          17,
          23
        ],
        "Gemma 4 E2B-it": [
          3,
          7,
          11,
          15
        ]
      },
      "uniform": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "InternVL3.5-2B-HF": [
          5,
          11,
          17,
          23
        ],
        "Gemma 4 E2B-it": [
          3,
          7,
          11,
          15
        ]
      },
      "selection_evidence": "V8_a70831c15da07ad465",
      "tolerance_pp": 5,
      "policy": {
        "version": "integrated_selection_v2_optional_ILA",
        "shortlist_structure": "one resident-model tap and one new-model unit when available",
        "max_units": 4,
        "beam_width": 2,
        "shortlist_per_parent": 2,
        "source_pool": "ILA optional peak, best two source gains and strongest evaluated singleton/model; at most4/model",
        "shortlist": "positive source gain * (1+CKA novelty) * (1+gradient residual novelty) * (1-negative-dot fraction) / predicted marginal seconds",
        "no_peak": "explicit source-gain fallback, never claim an ILA peak",
        "search": "FDA-inspired actual conditional attacks from measured strongest singleton and measured strongest pair; retain parents in cost Pareto beam; ILA not mandatory",
        "final_tolerance_pp": 5,
        "ablation_stage": 3,
        "final_target_feedback": false
      },
      "caveat": "Small dev selection, not statistical equivalence or proof of universal semantics; no confirmation/GPT feedback"
    },
    "internvl": {
      "primary": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "Qwen3.5-2B": [
          5,
          11,
          17,
          23
        ],
        "Gemma 4 E2B-it": [
          3,
          7,
          11,
          15
        ]
      },
      "uniform": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "Qwen3.5-2B": [
          5,
          11,
          17,
          23
        ],
        "Gemma 4 E2B-it": [
          3,
          7,
          11,
          15
        ]
      },
      "selection_evidence": "V8_3b89a4b7b99bc61fcb",
      "tolerance_pp": 5,
      "policy": {
        "version": "integrated_selection_v2_optional_ILA",
        "shortlist_structure": "one resident-model tap and one new-model unit when available",
        "max_units": 4,
        "beam_width": 2,
        "shortlist_per_parent": 2,
        "source_pool": "ILA optional peak, best two source gains and strongest evaluated singleton/model; at most4/model",
        "shortlist": "positive source gain * (1+CKA novelty) * (1+gradient residual novelty) * (1-negative-dot fraction) / predicted marginal seconds",
        "no_peak": "explicit source-gain fallback, never claim an ILA peak",
        "search": "FDA-inspired actual conditional attacks from measured strongest singleton and measured strongest pair; retain parents in cost Pareto beam; ILA not mandatory",
        "final_tolerance_pp": 5,
        "ablation_stage": 3,
        "final_target_feedback": false
      },
      "caveat": "Small dev selection, not statistical equivalence or proof of universal semantics; no confirmation/GPT feedback"
    },
    "gemma4": {
      "primary": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "Qwen3.5-2B": [
          5,
          11,
          17,
          23
        ],
        "InternVL3.5-2B-HF": [
          5,
          11,
          17,
          23
        ]
      },
      "uniform": {
        "CLIP ViT-L/14": [
          5,
          11,
          17,
          23
        ],
        "Qwen3.5-2B": [
          5,
          11,
          17,
          23
        ],
        "InternVL3.5-2B-HF": [
          5,
          11,
          17,
          23
        ]
      },
      "selection_evidence": "V8_c2efb191dbc9b217f3",
      "tolerance_pp": 5,
      "policy": {
        "version": "integrated_selection_v2_optional_ILA",
        "shortlist_structure": "one resident-model tap and one new-model unit when available",
        "max_units": 4,
        "beam_width": 2,
        "shortlist_per_parent": 2,
        "source_pool": "ILA optional peak, best two source gains and strongest evaluated singleton/model; at most4/model",
        "shortlist": "positive source gain * (1+CKA novelty) * (1+gradient residual novelty) * (1-negative-dot fraction) / predicted marginal seconds",
        "no_peak": "explicit source-gain fallback, never claim an ILA peak",
        "search": "FDA-inspired actual conditional attacks from measured strongest singleton and measured strongest pair; retain parents in cost Pareto beam; ILA not mandatory",
        "final_tolerance_pp": 5,
        "ablation_stage": 3,
        "final_target_feedback": false
      },
      "caveat": "Small dev selection, not statistical equivalence or proof of universal semantics; no confirmation/GPT feedback"
    }
  },
  "study_lock_sha256": "9d7f854f482881583446858d8c21193dfc8e77284bb39815ea02232f3517417d",
  "sources": {
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/combo_selection.py": "cf652fc6aea37a66f6ecf048700b8c1e35c58b13c40eea46f7d50665091b1642",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/watchdog.py": "8c61e0aac3a932528a012b18381a60b6f8360704e495ba2e957e18f3a4898c45",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/scheduler.py": "67eb3841c96d3015b9a3a5ca5cc33bb7968cdc8d66a464004b033579c22585f8",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/cli.py": "01f2e356379490d13c88c80f4bac9bc0c8f3f5bbe7d808b04c2e064cee214a37",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/reports.py": "5cee4cc10d622b0057212549a56010415eb4df389aaf651afa1f76b57b8e9ed7",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/ila_selection.py": "077f65d75a01d630de4d5ccaa75870acf352f0c445d6f9e508a30d9a9efd4692",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/dataset.py": "f8e622ee0fd634c6ab023ae99583b06c0e54eb1c0a407e85fc0c9775526a8209",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/selection.py": "d4064ab788e7922dc5bfd54b56d9a8dd2d5e8f33dc70556771dbd7ef9375aee3",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/resources.py": "4bb3359968abde5088b972b869df98788622e7c9c33bca01b759e980719aeceb",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/device_residency.py": "db0fb8ea027f11eb74556ae2c343a438eac6215de856ac18e291a00c9f0ae666",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/stage2_cli.py": "0a914abcd5a4badf83e149fa4f8eef8b78b441f0a633260ca8ccb59b7b336566",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/worker.py": "69a1473c13df3291e8779fda4a7c8e6385ea0b4e82aacf35fe44023710f52431",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/preflight.py": "df973aa5362247441db89dbbacd60fc57e2cf2408f257174ef80d3039c464088",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/common.py": "3260a13373777f608afdf18ed58bf36bc9f78f43b6ed4a3cdd80399696ef261d",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/geometry.py": "e0e17ed66e39df6e43348ea26e1e3873ff87397c10b13c0f0634c992915c86ec",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h/protocol.yaml": "49fc776bbbead2dc772fa09dfa53a7051cf8384f2e814d3868e023eedc4b8d5a",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/vision_features.py": "cf3b9b073cac4f8eaf7a5e4b81e679f58703f5e14885531aaf6dde1035af509b",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/pp4_objective.py": "8dae1f8fd3fcdfc649abe5158a85743153156ef342741d4a304b3a9cf96d0fbe",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/representation_objectives.py": "15739dc6bc568f4f5cac62628aac51333e8f36ce96c014c9815925484ef21c73",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/attack_optimizers.py": "745bd7bd5670441ea8e7d46aa09454a737d07a681ef2f609bb0e53428294b899",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/supplement_registry.py": "9054649575322bd1af6e0ab95024cdaf776f1b86ef5ea8f4e6cd84b3f9858f5f",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/run_codex_cli_pilot.py": "1c1d51f4bdb78412091605816339d1d062ef28375d67c9ea0912c4384b6e4054",
    "experiments/2026-10-a100-precommercial-v7-targeted-50/src/family_protocol.py": "7529fa4a4f8cb39349b0063e4e96b85d856c19e1e93dd9c8f4d2863df110afe9"
  },
  "commercial_family_rule": "Predeclared qwen-dev primary; GPT feedback forbidden",
  "stage2_primary_family": "qwen",
  "stage2_methods": [
    "selected",
    "uniform"
  ],
  "stage2_cohort_sha256": "0bd56026d80f7c6926409d31cd70078c15310b357b65e32d88a84be61473fc68",
  "stage2_model": "gpt-6.1-sol",
  "effort": "low",
  "approval": "user immediate start plus integrated-method-first override 2026-10-21"
}

## GPT confirmation
PENDING: no calls/results claimed.

## Interpretation and Stage3 discussion
No claim of universal visual head, proven causal mechanism or first multi-model/multi-layer attack. CKA/RSA and gradient conflict are diagnostics, not success surrogates. An exact implementation combination alone is not a novelty claim.

Must discuss: held-out categories and COCO original benchmark protocols; faithful vs adapted FDA/FOA/FTM/AdaEA/SEA baselines; multi-seed42/43/44; JPEG/resizing robustness; ten-class targeted ASR vs semantic change; nested surrogate selection and model family seen during development; independent final targets; selection cost and amortization.

All observed results: reports/conditions.csv,per_image.csv,directions.csv. Exact methods and hypotheses remain pending until measurements complete. No paper conclusions are fabricated.
