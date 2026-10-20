#!/usr/bin/env bash
# Laya multilingual runs AFTER the 5% BERT grid (same 8 GB GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/laya-logs
log="outputs/laya-logs/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
night=$(ls -t outputs/night-logs/*.log | head -1)
echo "WAIT Laya $(date -Is) for $night"
while ! grep -q '^FINISHED ' "$night" 2>/dev/null; do
  if ! pgrep -f '^bash scripts/run_night.sh$' >/dev/null; then
    echo 'Night grid stopped before FINISHED; Laya not started.'
    exit 1
  fi
  sleep 30
done
if find outputs/e9-laya -name results.json -print -quit 2>/dev/null | grep -q . && \
   find outputs/e10-laya-gate -name results.json -print -quit 2>/dev/null | grep -q .; then
  echo 'Laya results exist; skipping.'
  exit 0
fi
USE_TF=0 .laya-venv/bin/python scripts/serve_laya.py &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  if curl --silent --fail http://127.0.0.1:8018/health >/dev/null; then break; fi
  if ! kill -0 "$server" 2>/dev/null; then echo 'Laya server failed to start'; exit 1; fi
  sleep 30
done
curl --silent --fail http://127.0.0.1:8018/health >/dev/null
if ! find outputs/e9-laya -name results.json -print -quit 2>/dev/null | grep -q .; then
  uv run python -m scripts.eval_laya --url http://127.0.0.1:8018
fi
# Laya é o gate de dificuldade, BERT é o learner; piloto pequeno para caber nos 8GB.
if ! find outputs/e10-laya-gate -name results.json -print -quit 2>/dev/null | grep -q .; then
  uv run python main.py --generator openai --model deepseek-chat \
    --csv dataset/pre-processed-imbalanced-5pct.csv --test-csv dataset/pre-processed-test.csv \
    --seed 42 --prompt-version v1 --run-group e10-laya-gate --evaluator laya \
    --laya-url http://127.0.0.1:8018 --epochs 2 --batch-size 16 --top-k 4 \
    --learner-epochs 1 --learner-batch-size 4 --max-length 128 --max-run-tokens 100000
fi
printf 'LAYA FINISHED %s\n' "$(date -Is)"
