#!/usr/bin/env bash
# Orchestrates baseline + ablation runs after the main FAILGROUND run finishes.
# IC and ALFWorld run in parallel within each stage; stages run sequentially.
set -uo pipefail

cd "$(dirname "$0")"
source /home/taher/Taher_Codebase/VATRAG/venv/bin/activate
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_IC_LOG="${1:-logs/ic_r4.log}"
MAIN_ALF_LOG="${2:-logs/alf_r4.log}"

echo "[pipeline] waiting for main run to finish: $MAIN_IC_LOG / $MAIN_ALF_LOG"
until grep -q "Results ->" "$MAIN_IC_LOG" 2>/dev/null && grep -q "Results ->" "$MAIN_ALF_LOG" 2>/dev/null; do
  sleep 30
done
echo "[pipeline] main run done. Starting baseline (parallel)."

python3 run_experiment.py --benchmark intercode --method baseline --episodes 50 > "$LOG_DIR/ic_baseline.log" 2>&1 &
PID_IC=$!
python3 run_experiment.py --benchmark alfworld --method baseline --episodes 50 > "$LOG_DIR/alf_baseline.log" 2>&1 &
PID_ALF=$!
wait $PID_IC $PID_ALF
echo "[pipeline] baseline done. Starting ablations (parallel within each, sequential across)."

for AB in A1 A2 A3 A4 A5 A6; do
  echo "[pipeline] === ablation $AB ==="
  python3 run_experiment.py --benchmark intercode --ablation "$AB" --episodes 25 > "$LOG_DIR/ic_${AB}.log" 2>&1 &
  PID_IC=$!
  python3 run_experiment.py --benchmark alfworld --ablation "$AB" --episodes 25 > "$LOG_DIR/alf_${AB}.log" 2>&1 &
  PID_ALF=$!
  wait $PID_IC $PID_ALF
  echo "[pipeline] === ablation $AB done ==="
done

echo "[pipeline] ALL DONE"
ls -la results/
