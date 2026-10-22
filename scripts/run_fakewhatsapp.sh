#!/usr/bin/env bash
# Complete, resumable FakeWhatsApp.Br temporal grid. Never writes to E1-E10.
set +e
cd "$(dirname "$0")/.."
mkdir -p outputs/fakewhatsapp-logs
log="outputs/fakewhatsapp-logs/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1

echo "START $(date -Is) log=$log"
uv run python -m tests.test_core || exit 1
uv run python scripts/prepare_fakewhatsapp.py || exit 1

LLM_OK=1
uv run python - <<'PY' || LLM_OK=0
from src.config import Config
from src.generator import OpenAIGenerator
g=OpenAIGenerator(Config(openai_model='deepseek-chat'))
g.client.chat.completions.create(model='deepseek-chat',messages=[{'role':'user','content':'ok'}],max_completion_tokens=8)
PY
[[ "$LLM_OK" == 1 ]] || echo 'API indisponível: braços pagos serão pulados; Real continua.'

completed() {
  uv run python - "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
group,arm,seed=sys.argv[1:]
for f in (Path('outputs')/group).glob('*/results.json'):
 d=json.loads(f.read_text())
 if d.get('arm')==arm and d.get('config',{}).get('random_state')==seed:
  raise SystemExit(0)
raise SystemExit(1)
PY
}

run_one() {
  group="$1" csv="$2" arm="$3" seed="$4"
  if completed "$group" "$arm" "$seed"; then echo "SKIP complete $group $arm s$seed"; return; fi
  if [[ "$arm" != real && "$LLM_OK" == 0 ]]; then echo "SKIP API $group $arm s$seed"; return; fi
  common=(--csv "$csv" --val-csv dataset/fakewhatsapp/validation-september.csv --test-csv dataset/fakewhatsapp/test-october.csv --run-group "$group" --seed "$seed")
  gen=(--generator openai --model deepseek-chat --prompt-version v3 --gen-max-tokens 256 --generator-workers 4 --max-run-tokens 100000)
  echo "RUN $(date -Is) $group $arm s$seed"
  if [[ "$arm" == real ]]; then
    uv run python baseline.py --arm real "${common[@]}"
  elif [[ "$arm" == static ]]; then
    uv run python baseline.py --arm static --n-samples 500 "${common[@]}" "${gen[@]}"
  else
    uv run python main.py --epochs 5 --batch-size 64 --top-k 16 "${common[@]}" "${gen[@]}"
  fi
  code=$?
  echo "DONE exit=$code $(date -Is) $group $arm s$seed"
}

for spec in \
  'fakewhatsapp-full dataset/fakewhatsapp/train-full.csv' \
  'fakewhatsapp-10pct dataset/fakewhatsapp/train-10pct.csv' \
  'fakewhatsapp-5pct dataset/fakewhatsapp/train-5pct.csv'; do
  read -r group csv <<< "$spec"
  for seed in 42 43 44; do run_one "$group" "$csv" real "$seed"; done
  for seed in 42 43 44; do run_one "$group" "$csv" static "$seed"; done
  for seed in 42 43 44; do run_one "$group" "$csv" ai "$seed"; done
done

uv run python scripts/analyze_fakewhatsapp.py
echo "FINISHED $(date -Is)"
