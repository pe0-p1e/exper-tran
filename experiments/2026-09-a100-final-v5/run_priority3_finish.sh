#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$ROOT/data/imagenet_diverse10_minimal}"
export PYTHONPATH="$ROOT/src:$ROOT/experiments/2026-09-a100-final-v5/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

PYTHON="$ROOT/.venv-a100/bin/python"
EXPERIMENT="$ROOT/experiments/2026-09-a100-final-v5"
CONFIG="$EXPERIMENT/config/full90_selected_layer_priority3.yaml"
OUTPUT="$ROOT/outputs/a100_final_v5"
LOG_DIR="$OUTPUT/logs/priority3_finish"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/priority3_finish_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "waiting_for_three_pair_evaluations_and_embeddings time=$(date -u +%FT%TZ)"
while ! "$PYTHON" - <<'PY'
import json
from pathlib import Path
import yaml
root = Path('outputs/a100_final_v5')
config = yaml.safe_load(Path('experiments/2026-09-a100-final-v5/config/full90_selected_layer_priority3.yaml').read_text())
plot_ids = [item['id'] for item in config['transitions'] if int(item['source']) < int(item['target'])]
for pair in ('P02', 'P14', 'P23'):
    states = list((root / 'states_layer_100' / pair).glob('T*/batch_00.json'))
    if len(states) != 90 or any(json.loads(path.read_text()).get('status') != 'complete' for path in states):
        raise SystemExit(1)
    analysis = root / 'analysis/single_proxy_full90_priority3' / pair
    if len(plot_ids) != 45 or not all(
        (analysis / kind / f'{pair}_{transition_id}.png').is_file()
        for transition_id in plot_ids for kind in ('pca', 'tsne')
    ):
        raise SystemExit(1)
PY
do
  sleep 30
done

echo "three_pair_evaluation_complete time=$(date -u +%FT%TZ)"
"$PYTHON" - <<'PY'
from pathlib import Path
from common import load_experiment
from run_a100_final import summarize
config = load_experiment(Path('experiments/2026-09-a100-final-v5/config/full90_selected_layer_priority3.yaml'))
summarize(Path('outputs/a100_final_v5'), config)
PY
"$PYTHON" "$EXPERIMENT/src/split_full90_summary.py" \
  --config "$CONFIG" \
  --summary "$OUTPUT/summaries/single_proxy_full90_priority3.csv" \
  --output-dir "$OUTPUT/summaries"

for pair in P02 P14 P23; do
  echo "finish_full90_geometry pair=$pair time=$(date -u +%FT%TZ)"
  "$PYTHON" "$EXPERIMENT/src/analyze_joint_geometry.py" \
    --config "$CONFIG" --output-dir "$OUTPUT" --pair "$pair" \
    --batch-size 50 --plot-transition T01 \
    --analysis-subdir "single_proxy_full90_priority3/$pair" --resume
done

"$PYTHON" "$EXPERIMENT/src/make_priority3_examples.py"
"$PYTHON" "$EXPERIMENT/src/analyze_all_completed.py"
"$PYTHON" "$EXPERIMENT/src/export_all_measurements.py"
"$PYTHON" "$EXPERIMENT/src/package_priority3_results.py"
"$PYTHON" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
path = Path('outputs/a100_final_v5/shareable/priority3/a100_final_v5_priority3_results.zip')
with ZipFile(path) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f'Corrupt archive member: {bad}')
    print(f'archive_verified members={len(archive.namelist())} bytes={path.stat().st_size}')
PY
echo "priority3_finish_done time=$(date -u +%FT%TZ)"
