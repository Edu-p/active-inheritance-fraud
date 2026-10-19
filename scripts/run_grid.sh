#!/usr/bin/env bash
# Roda a grade completa E1-E5 do zero. Continua em caso de erro por run (set +e).
set +e
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d-%H%M%S)
LOG="outputs/grid_${TS}.log"
mkdir -p outputs
echo "grid start $(date) — log: $LOG"

run() {
  label="$1"; shift
  echo "### $label $(date)" >> "$LOG"
  uv run "$@" >> "$LOG" 2>&1
  echo "### exit=$? $label $(date)" >> "$LOG"
}

MODEL=deepseek-chat   # deepseek-flash é reasoning e recusa fake news — usar chat

for s in 42 43 44; do
  run "E1 real s$s" python baseline.py --arm real --seed "$s"
done

for s in 42 43 44; do
  for n in 32 79 158; do
    run "E2 static n$n s$s" python baseline.py --arm static --generator openai --model "$MODEL" --n-samples "$n" --seed "$s"
  done
done

for s in 42 43 44; do
  run "E3 ai s$s" python main.py --generator openai --model "$MODEL" --seed "$s"
done

run "E4 ai pesos" python main.py --generator openai --model "$MODEL" --w-error 0.7 --w-uncertainty 0.2 --w-diversity 0.1

run "E5 ai topk8" python main.py --generator openai --model "$MODEL" --top-k 8

echo "grid done $(date)" >> "$LOG"
