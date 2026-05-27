#!/usr/bin/env bash
# Run the full paper experiment suite. WARNING: this is heavy (GPU-days with the
# 7B model and full GCG). Override knobs via env vars, e.g.:
#   MODEL=Qwen/Qwen2.5-7B-Instruct DEVICE=cuda bash scripts/run_all.sh
#   QUICK=1 bash scripts/run_all.sh         # fewer rounds/steps for a fast pass
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-auto}"
PY="${PY:-python}"
EXTRA=""
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA="--rounds 30 --num-questions 3"
  echo "[run_all] QUICK mode: $EXTRA"
fi
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA="$EXTRA --load-in-4bit"
  echo "[run_all] 4-bit quantization enabled"
fi

run() { $PY scripts/run_experiment.py --model "$MODEL" --device "$DEVICE" --reuse-suffix $EXTRA "$@"; }

DENSITIES=(0.01 0.5 0.99)
ATTACKS=(clean gcg arcj)

# ---- Structure experiment (Table 1): graph / line / star, 20 agents ----------
for atk in "${ATTACKS[@]}"; do
  for topo in graph line star; do
    for d in "${DENSITIES[@]}"; do
      run --config "configs/structure_${topo}.yaml" --attack "$atk" --density "$d" \
          --name "structure_${topo}_${atk}_d${d}"
    done
  done
done

# ---- Scale experiment (Table 2): 6 / 20 / 100 agents, graph ------------------
for atk in "${ATTACKS[@]}"; do
  for n in 6 20 100; do
    for d in "${DENSITIES[@]}"; do
      run --config "configs/scale_${n}.yaml" --attack "$atk" --density "$d" \
          --name "scale_${n}_${atk}_d${d}"
    done
  done
done

# ---- Toxicity-disappearing experiment (Fig. 4 / 7) ---------------------------
$PY scripts/run_toxicity.py --config configs/toxicity_disappearing.yaml \
    --model "$MODEL" --device "$DEVICE" --out results/toxicity.json

# ---- Aggregate plots & tables ------------------------------------------------
echo "==== Table 1 (structure) ===="
$PY scripts/plot_results.py table --inputs results/structure_*_d*.json || true
echo "==== Table 2 (scale) ===="
$PY scripts/plot_results.py table --inputs results/scale_*_d*.json || true
$PY scripts/plot_results.py toxicity --input results/toxicity.json --out results/toxicity.png || true
echo "[run_all] done. Results in results/"
