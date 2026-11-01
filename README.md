# Active-Inheritance Guided Synthetic Generation 🧬🤖

**Repo:** https://github.com/Edu-p/active-inheritance-fraud

Implementa **Herança Ativa (ActiveInheritance)**: loop cíclico em que um LLM gera
textos `fake` e um avaliador adversarial pontua cada candidato por
`score = 0.4·erro + 0.4·incerteza + 0.2·diversidade`, herdando só o top-k no
retreino do classificador downstream (BERTimbau).

Baseline: [profile2gen](https://github.com/szanara/profile2gen) (Maynara et al., ICML 2025).

📄 **Artigo atual:** [`report-latex/paper-active-inheritance-multidomain.pdf`](report-latex/paper-active-inheritance-multidomain.pdf)
(estudo controlado 4 domínios × 10 seeds). Vault de estado: `**__active_obsidian/`.

---

## Resultado central (estudo controlado, n=10)

4 regras de herança (`All` / `Random-k` / `Score-k` / `Easy-k`) sobre os MESMOS
candidatos zero-shot congelados, 4 domínios PT-BR (Fake.br, FakeWhatsApp.Br,
COVID19.BR, FACTCK.BR) × 3 escassezes × 10 seeds = **440 runs**.

- **O score NÃO domina o sorteio**: `score-k ≈ random-k` na maioria (nenhuma
  célula significativa após Holm).
- **Curadoria de volume é o efeito robusto**: `All` colapsa sob escassez
  (FACTCK 10%: 0.840 vs 0.94 curado).
- **Ajuda de verdade só onde o learner é competente e a tarefa é difícil**
  (COVID19.BR 10%: score-k 0.728 vs random-k 0.655, p=0.084 raw / 0.92 Holm).
- **Atalho estilístico**: um TF-IDF+LR separa sintético de real com AUC=1.0
  (comprimento em FACTCK, digit_ratio no Fake.br, deriva semântica no WhatsApp).

Consolidação: `uv run python scripts/analyze_normalized.py` (+ `--latex`).
Significância: `uv run python scripts/significance.py`.

---

## Pipeline

1. **Data** — 4 domínios PT-BR + LIAR e FakeNewsNet (EN), com escassez full/10%/5%.
2. **Gerador** — DeepSeek `deepseek-chat` ou OpenRouter (GPT/Llama/Mistral/Gemma/Claude).
3. **Learner** — BERTimbau (`neuralmind/bert-base-portuguese-cased`).
4. **Avaliador** — `score = w_e·erro + w_u·incerteza + w_d·diversidade`.
5. **Herança** — top-k → pool → retreina → próximo ciclo.

---

## Install

```bash
git clone https://github.com/Edu-p/active-inheritance-fraud.git
cd active-inheritance-fraud
uv sync              # torch cu124 no pyproject (GPU)
cp .env.example .env # edite a chave
```

`.env` (DeepSeek, padrão):

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat   # NÃO use deepseek-flash (reasoning, recusa fake news)
```

OpenRouter (multi-provider): adicione `OPEN_ROUTER_API_KEY=sk-or-...` e passe
`--base-url https://openrouter.ai/api/v1 --model <id>` (ex.: `openai/gpt-6.1-terra`,
`anthropic/claude-haiku-5.5`).

---

## Usage

```bash
# datasets escassos (uma vez; LIAR/FakeNewsNet = EN)
uv run python scripts/make_imbalanced.py
uv run python scripts/prepare_liar.py
uv run python scripts/prepare_fakenewsnet.py

# estudo controlado (gerar candidatos = API; replay = GPU, resumível)
uv run python scripts/controlled_curatorship.py --generate --scenario covid19br-10pct --seed 42
uv run python scripts/controlled_curatorship.py --replay --candidates <candidates.json> --arm score-k

# pipeline completo PT (resumível)
bash scripts/run_normalize_all.sh

# ablações / generalização (tiers NeurIPS, resumíveis)
bash scripts/run_neurotier.sh                    # styleblind + weights + multilearner + competence + band
bash scripts/run_multigenerator.sh               # T2.1 multi-LLM (API OpenRouter)
bash scripts/run_english_gen.sh                  # T3.1 LIAR + FakeNewsNet (API DeepSeek)

# análise
uv run python scripts/analyze_normalized.py      # tabelas do estudo controlado
uv run python scripts/analyze_ablations.py       # pesos + style-blind
uv run python scripts/style_gap.py               # N1: atalho estilístico
uv run python scripts/significance.py            # T1.3: Wilcoxon + Holm

# self-check (sem torch)
uv run python -m tests.test_core
```

## Outputs

```
outputs/
├── revision-controlled-<scenario>-<prompt>/  # estudo controlado (all/random-k/score-k/easy-k)
├── ablation-weights-*/   # T1.2 (frozen scorer)
├── ablation-styleblind-*/# N2  (SemanticProbe scorer)
├── multilearner-*/       # T2.2
├── multigenerator-*/     # T2.1
├── style-gap/            # N1 (JSON por cenário)
└── revision-candidates/  # candidates.json congelados (fonte dos replays)
```

Cada run: `outputs/<group>/<arm>-s<seed>-<timestamp>/results.json` (imutável).

## Main arguments (`main.py` / `controlled_curatorship.py`)

| Argument | Default | Descrição |
|---|---|---|
| `--scenario` | — | `fakewhatsapp-full` · `covid19br-10pct` · `factckbr-5pct` · `liar-full` · `fakenewsnet-10pct` … |
| `--arm` | — | `all` \| `random-k` \| `score-k` \| `easy-k` |
| `--top-k` | `16` | herdados/ciclo |
| `--model` | `deepseek-chat` | LLM gerador (ou id OpenRouter) |
| `--base-url` | env | endpoint OpenAI-compatible |
| `--learner-model` | `neuralmind/bert-base-portuguese-cased` | learner |
| `--seed` | `42` | semente |
