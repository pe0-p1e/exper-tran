#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="$REPO_ROOT/experiments/2026-10-a100-precommercial-v6"
OUT="$REPO_ROOT/outputs/a100_precommercial_v6_multi_repr_recovery"
PY="$REPO_ROOT/.venv-a100/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PYTHONPATH="$REPO_ROOT/src:$EXP_ROOT/src:$REPO_ROOT/experiments/2026-09-a100-final-v5/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PRIMARY_ML_CKA_STRICT_BF16=1
mkdir -p "$OUT/logs/stages" "$OUT/audits" "$HF_HOME"
if [[ ! -f "$OUT/audits/campaign_started_epoch" ]]; then date +%s > "$OUT/audits/campaign_started_epoch"; fi
exec > >(tee -a "$OUT/logs/run_multiple_proxy_recovery.log") 2>&1

stage() {
  local name="$1"; shift
  echo "[$(date -Is)] BEGIN $name"
  "$@" 2>&1 | tee -a "$OUT/logs/stages/${name}.log"
  echo "[$(date -Is)] END $name"
}

echo "[recovery] branch=$(git -C "$REPO_ROOT" branch --show-current) head=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "experiment/a100-precommercial-v6" ]] || { echo "Wrong branch; expected experiment/a100-precommercial-v6"; exit 2; }
[[ -x "$PY" ]] || { echo "Missing venv python: $PY"; exit 2; }
nvidia-smi
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert "A100" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
assert torch.cuda.is_bf16_supported(), "BF16 unsupported"
print(f"preflight: {torch.__version__=} {torch.version.cuda=} {torch.cuda.get_device_name(0)=} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
PY
df -h "$REPO_ROOT" "$HF_HOME"

stage 02_dataset_prepare "$PY" "$EXP_ROOT/src/prepare_recovery_data.py" --output-dir "$OUT"
stage 03_model_download "$PY" "$EXP_ROOT/src/download_models.py" --output-dir "$OUT" --cache "$HF_HOME"

# A complete passing revision-matched audit is reusable on resume. A missing,
# partial, or stale audit forces a fresh load-and-check before inference.
if ! "$PY" "$EXP_ROOT/src/check_recovery_audit.py" --output-dir "$OUT" >/dev/null 2>&1; then
  stage 01_model_audit "$PY" "$EXP_ROOT/src/audit_models.py" --output-dir "$OUT"
else
  echo "[model audit] all pinned model/role audits already pass; reusing immutable audit records"
fi

stage 04_gradient_calibration "$PY" "$EXP_ROOT/src/calibrate_recovery.py" --output-dir "$OUT"

# End-to-end two-image, two-step heterogeneous ensemble smoke; repeat once to
# exercise the runner's completed-batch resume path. Smoke data is namespaced.
if ! "$PY" "$EXP_ROOT/src/check_smoke_artifacts.py" --output-dir "$OUT" >/dev/null 2>&1; then
  stage smoke_first "$PY" "$EXP_ROOT/src/run_recovery_ensemble.py" --ensemble-id smoke_qwen2b_clip_dino --target "Qwen3.5-27B" --proxies "Qwen3.5-2B" "CLIP ViT-L/14" "DINOv2-Large" --transitions T01 --batch-size 2 --smoke --output-dir "$OUT"
  stage smoke_resume "$PY" "$EXP_ROOT/src/run_recovery_ensemble.py" --ensemble-id smoke_qwen2b_clip_dino --target "Qwen3.5-27B" --proxies "Qwen3.5-2B" "CLIP ViT-L/14" "DINOv2-Large" --transitions T01 --batch-size 2 --smoke --output-dir "$OUT"
fi
if ! "$PY" "$EXP_ROOT/src/check_frozen_target_smoke.py" --output-dir "$OUT" >/dev/null 2>&1; then
  stage target_smoke "$PY" "$EXP_ROOT/src/smoke_frozen_target_generation.py" --output-dir "$OUT"
fi

stage 06_multiple_proxy_attacks "$PY" "$EXP_ROOT/src/run_recovery_campaign.py" attacks --batch-size 2
stage 07_frozen_png_target_evaluation "$PY" "$EXP_ROOT/src/run_recovery_campaign.py" evaluate
stage 08_representation_analysis "$PY" "$EXP_ROOT/src/analyze_recovery.py" --output-dir "$OUT"
stage 09_validation_and_report "$PY" "$EXP_ROOT/src/finalize_recovery.py" --output-dir "$OUT"

echo "[$(date -Is)] recovery launcher finished; inspect $OUT/reports/multiple_proxy_recovery_report.md"
