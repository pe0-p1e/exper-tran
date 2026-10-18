#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Ten-hour research interrupted; checkpoint retained"; exit 130' INT TERM
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/ephemeral/a100_precommercial_v6/model-cache/hf-cache}"
export PRIMARY_ML_CKA_STRICT_BF16=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/outputs/a100_precommercial_v7_targeted50"
mkdir -p "$OUT/ten_hour_research"
exec 9>"$OUT/campaign.lock"
flock -n 9 || { echo 'Another V7 campaign is active'; exit 2; }
exec > >(tee -a "$OUT/ten_hour_research/run.log") 2>&1
PY="$ROOT/.venv-a100/bin/python"
"$PY" -m pytest -q experiments/2026-10-a100-precommercial-v7-targeted-50/tests
REMAINING="$($PY - <<'PY'
import json,time,pathlib,sys,os
sys.path.insert(0,'experiments/2026-10-a100-precommercial-v7-targeted-50/src')
from extended_research_plan import extended_budget
from v7_common import atomic_json
p=pathlib.Path('outputs/a100_precommercial_v7_targeted50/ten_hour_research/budget.json')
old=json.loads(p.read_text()) if p.exists() else None
# User-approved total 20-hour budget, measured from the original campaign start.
# Repeated resumes do not add another 20 hours or shorten a larger prior budget.
budget=extended_budget(old,float(os.environ.get('V7_RESEARCH_HOURS','20')),time.time())
lock=p.parent/'selected_method_lock.json'
if old and lock.exists() and budget['deadline']!=old['deadline']:
    raise RuntimeError('Cannot extend adaptive research after final method lock')
atomic_json(p,budget)
print(max(1,int(budget['deadline']-time.time())))
PY
)"
set +e
timeout --signal=INT --kill-after=120 "$REMAINING" "$PY" -u experiments/2026-10-a100-precommercial-v7-targeted-50/src/ten_hour_research.py
STATUS=$?
set -e
"$PY" experiments/2026-10-a100-precommercial-v7-targeted-50/src/finalize_ten_hour.py --exit-code "$STATUS"
exit "$STATUS"
