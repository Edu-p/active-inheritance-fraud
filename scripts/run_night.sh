#!/usr/bin/env bash
# Follow-up isolado: nunca executa a grade antiga nem reescreve results.json.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/night-logs
log="outputs/night-logs/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1

echo "START $(date -Is) log=$log"
uv run python -m tests.test_core
uv run python scripts/make_imbalanced.py --frac .05 --out-train dataset/pre-processed-imbalanced-5pct.csv --out-test dataset/pre-processed-test.csv

# Uma única chamada barata: credencial inválida/desativada não deve interromper E6 Real.
LLM_OK=1
if ! uv run python - <<'PY'
from src.config import Config
from src.generator import OpenAIGenerator
g = OpenAIGenerator(Config(openai_model='deepseek-chat'))
g.client.chat.completions.create(model='deepseek-chat', messages=[{'role':'user','content':'ok'}], max_completion_tokens=8)
PY
then
  LLM_OK=0
  echo "SKIP geração DeepSeek: credencial/serviço indisponível. Real e probe Real continuam."
fi

run() {
  local group="$1" arm="$2" seed="$3" n="$4" prompt="$5" kind="$6" model="$7" url="$8"
  if [[ "$arm" != real && "$url" == - && "$LLM_OK" == 0 ]]; then
    echo "SKIP $group $arm s$seed: API DeepSeek indisponível"
    return
  fi
  # Retomada: verifica metadados do JSON, não apenas diretório. Nunca duplica uma run concluída.
  if uv run python - "$group" "$arm" "$seed" "$n" "$prompt" "$kind" "$model" <<'PY'
import sys, json
from pathlib import Path
group, arm, seed, n, prompt, kind, model = sys.argv[1:]
for f in (Path('outputs') / group).glob('*/results.json'):
    d = json.loads(f.read_text()); c = d['config']
    if (d['arm'], c['random_state'], c.get('n_samples', 'None'), c.get('prompt_version', 'v1'), c.get('learner_kind', 'bert'), c['openai_model']) == (arm, seed, n, prompt, kind, model):
        sys.exit(0)
sys.exit(1)
PY
  then echo "SKIP completed $group $arm s$seed n$n $prompt $kind $model"; return; fi

  local cap=800000
  if [[ "$url" != - ]]; then cap=100000; fi  # GPT: no máximo ~100k tokens nesta comparação pequena.
  local common=(--csv dataset/pre-processed-imbalanced-5pct.csv --test-csv dataset/pre-processed-test.csv --seed "$seed" --run-group "$group" --learner-kind "$kind" --prompt-version "$prompt" --model "$model" --max-run-tokens "$cap")
  local provider=(--generator openai)
  if [[ "$url" != "-" ]]; then provider+=(--base-url "$url"); fi
  echo "RUN $(date -Is) $group $arm s$seed n$n $prompt $kind $model"
  if [[ "$arm" == ai ]]; then
    uv run python main.py "${common[@]}" "${provider[@]}"
  elif [[ "$arm" == real ]]; then
    uv run python baseline.py --arm real "${common[@]}"
  else
    uv run python baseline.py --arm static --n-samples "$n" "${common[@]}" "${provider[@]}"
  fi
  echo "DONE $(date -Is) $group $arm s$seed"
}

# E6: única mudança em relação a E1/E2/E3 é o treino 5%; prompt v1, BERT e DeepSeek-chat iguais.
for seed in 42 43 44; do
  run e6-5pct real "$seed" 32 v1 bert deepseek-chat -
  run e6-5pct static "$seed" 158 v1 bert deepseek-chat -
  run e6-5pct ai "$seed" None v1 bert deepseek-chat -
done

# E7: estudo exploratório (prompt v2 + decisão por probe semântico). NÃO isola o efeito do learner.
run e7-probe real 42 32 v2 semantic-probe deepseek-chat -
run e7-probe static 42 158 v2 semantic-probe deepseek-chat -
run e7-probe ai 42 None v2 semantic-probe deepseek-chat -

# E8: mesma configuração para dois geradores com prompt v2; GPT só com credencial separada.
run e8-generators static 42 32 v2 semantic-probe deepseek-chat -
if uv run python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv('.env')
raise SystemExit(0 if os.getenv('GPT_API_KEY') else 1)
PY
then
  run e8-generators static 42 32 v2 semantic-probe gpt-4o-mini https://api.openai.com/v1
else
  echo "SKIP GPT: GPT_API_KEY ausente; nunca enviar credencial DeepSeek à API OpenAI."
fi
echo "FINISHED $(date -Is)"
