# Active-Inheritance Guided Synthetic Generation 🧬🤖

**Repo:** https://github.com/Edu-p/active-inheritance-fraud

Implements **Active Inheritance**: a cyclic LLM generation loop for fraudulent-text detection, guided by an **adversarial evaluator** that scores synthetic samples by the downstream classifier's *error + uncertainty + diversity*, inheriting only the top-k (decision boundary) into the next cycle.

Baseline: [profile2gen](https://github.com/szanara/profile2gen) (Maynara et al., ICML 2025).

📄 **Análise consolidada E1–E10:** [`docs/analise_completa_experimentos.md`](docs/analise_completa_experimentos.md)

Outros docs: [`docs/implementation.md`](docs/implementation.md) · [`planning/PLAN.md`](planning/PLAN.md) · [`planning/RUN.md`](planning/RUN.md) · [`planning/HANDOFF.md`](planning/HANDOFF.md) · [Seguimento noturno](planning/FOLLOWUP_NIGHT.md)

> Seguimento (separado da tabela abaixo): `bash scripts/run_night.sh` cria `outputs/e6-5pct/`, `e7-probe/`, `e8-generators/` sem tocar em `e1`–`e5`. Consulte `planning/FOLLOWUP_NIGHT.md` para status, limites e credenciais indisponíveis.

---

## Results (Fake.br-Corpus, cenário de escassez: 10% fake)

| Braço | F1 (teste) | tokens/run |
|---|---|---|
| Real (sem sintético) | 0.887 ± 0.026 | 0 |
| Static N=32 | 0.789 ± 0.113 | ~20k |
| Static N=79 | 0.852 ± 0.034 | ~53k |
| Static N=158 | 0.916 ± 0.013 | ~103k |
| **ActiveInheritance** | **0.924 ± 0.007** | ~645k |

> ActiveInheritance é o mais consistente (menor desvio entre seeds) e atinge o topo de F1, mas **não** supera o Static em eficiência de tokens na config atual (batch 64 × 5 ciclos). Análise completa em `report-latex/main.pdf`.

---

## Pipeline

1. **Data 0** — Fake.br-Corpus: treino escasso (10% fake) + holdout balanceado.
2. **LLM generator** — DeepSeek `deepseek-chat`, condicionado ao pool herdado.
3. **Learner** — BERTimbau (`neuralmind/bert-base-portuguese-cased`).
4. **Adversarial evaluator** — `score = w1·erro + w2·incerteza + w3·diversidade`.
5. **Active inheritance** — top-k → pool → retreina → próximo ciclo.

---

## Install

```bash
git clone https://github.com/Edu-p/active-inheritance-fraud.git
cd active-inheritance-fraud
uv sync              # torch cu124 já configurado no pyproject (GPU)
cp .env.example .env # edite a chave
```

`.env`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat   # NÃO use deepseek-flash (reasoning, recusa fake news)
```

---

## Usage

```bash
# 1. dataset escasso + holdout (uma vez só)
uv run python scripts/make_imbalanced.py

# 2. braços
uv run python baseline.py --arm real --seed 42
uv run python baseline.py --arm static --generator openai --model deepseek-chat --n-samples 158 --seed 42
uv run python main.py --generator openai --model deepseek-chat --seed 42

# 3. grade completa E1–E5 (background)
nohup bash scripts/run_grid.sh > outputs/grid_console.log 2>&1 &

# 4. figuras + estatística
uv run python plots.py
uv run python scripts/stats.py

# 5. self-check (sem torch)
uv run python -m tests.test_core
```

## Outputs

```
outputs/
├── e1/   # Real
├── e2/   # Static
├── e3/   # ActiveInheritance
├── e4/   # ablação: pesos
├── e5/   # ablação: top-k
└── sandbox/
```

Cada run: `outputs/<exp>/<arm>-s<seed>-<timestamp>/results.json`.

## Main arguments (`main.py`)

| Argument | Default | Descrição |
|---|---|---|
| `--generator` | `mock` | `mock` \| `openai` |
| `--epochs` | `5` | ciclos T |
| `--batch-size` | `64` | lote sintético/ciclo |
| `--top-k` | `16` | herdados/ciclo |
| `--model` | `deepseek-chat` | LLM |
| `--learner-model` | `neuralmind/bert-base-portuguese-cased` | learner |
| `--embedder` | `semantic` | `semantic` \| `tfidf` |
| `--seed` | `42` | semente |
