#!/usr/bin/env bash
# Run all FAILGROUND ablation studies after main experiments complete.
# Usage: bash run_ablations.sh [bench] [episodes]

set -euo pipefail

BENCH=${1:-intercode}
EPISODES=${2:-50}
VENV=/home/taher/Taher_Codebase/VATRAG/venv
LOG_DIR=/tmp/failground_ablation_logs

source "$VENV/bin/activate"
mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY not set" >&2
  exit 1
fi

cd "$(dirname "$0")"

for ablation in A4 A5 A1 A2 A3 A6; do
  outfile="results/${BENCH}_${ablation}.json"
  if [ -f "$outfile" ]; then
    echo "SKIP $ablation — $outfile already exists"
    continue
  fi
  echo "=== Running ablation $ablation on $BENCH ($EPISODES episodes) ==="
  python run_experiment.py \
    --benchmark "$BENCH" \
    --ablation "$ablation" \
    --episodes "$EPISODES" \
    2>&1 | tee "$LOG_DIR/${BENCH}_${ablation}.log"
  echo "--- Done: $ablation ---"
done

echo ""
echo "All ablations complete. Results in results/"
ls -lh results/
