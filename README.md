# Geração Sintética Guiada por Herança Ativa 🧬🤖

Este repositório contém a implementação do *framework* de **Herança Ativa** focado na geração de dados sintéticos para tarefas de cibersegurança e detecção de textos fraudulentos (ex: Phishing, Fake News, Anúncios Falsos).

Em vez de gerar dados sintéticos de forma estática e passiva (o que frequentemente leva ao colapso do modelo), este pipeline utiliza um **Avaliador Adversarial** guiado por um modelo *learner* (BERT). Apenas as amostras sintéticas que maximizam a incerteza, o erro do modelo atual e a diversidade do pool são "herdadas" para o próximo ciclo de treinamento, atuando como *prática deliberada*.

📄 **Documentação Completa:** [`docs/implementacao.md`](docs/implementacao.md)

---

## 🏗️ Arquitetura do Pipeline

1. **Data 0 (Dados Reais):** O conjunto inicial (ex: *Fake.br-Corpus*).
2. **Gerador Sintético:** LLMs (como DeepSeek) instruídos a gerar novos exemplos fraudulentos.
3. **Learner (BERT):** Um classificador *downstream* que está sendo treinado para detectar as fraudes.
4. **Avaliador Adversarial:** Filtra o lote gerado pelo LLM, calculando a entropia (incerteza) e o erro do Learner, somado à distância de cosseno (diversidade via TF-IDF/Embeddings).
5. **Herança Ativa:** O Top-K mais "desafiador" é selecionado, injetado no dataset de treino, e o ciclo reinicia.

---

## ⚙️ Instalação e Configuração

Recomenda-se o uso de [uv](https://docs.astral.sh/uv/) (Python 3.9+).

```bash
# Instala o uv (uma vez)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone o repositório
git clone https://github.com/seu-usuario/active-inheritance-fraud.git
cd active-inheritance-fraud

# Cria o venv (.venv) e instala as dependências do pyproject.toml
uv sync

# Opcional, sem GPU: troca o torch pelo build CPU-only (bem menor)
uv remove torch && uv add torch --index https://download.pytorch.org/whl/cpu
```

Para o gerador via LLM, copie `.env.example` para `.env` e preencha:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
```

---

## 🚀 Uso

```bash
# Modo mock (offline, sem LLM — perturba notícias fake reais)
uv run python main.py --generator mock --epochs 5 --top-k 16 --batch-size 64

# Modo LLM (endpoint OpenAI-compatible)
uv run python main.py --generator openai --model deepseek-flash \
  --base-url https://api.deepseek.com --epochs 5 --top-k 16

# Self-check do núcleo (sem torch)
uv run python tests/test_core.py
```

### Principais argumentos

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--generator` | `mock` | `mock` ou `openai` |
| `--epochs` | `5` | ciclos de Herança Ativa (T) |
| `--batch-size` | `64` | tamanho do lote sintético gerado por ciclo |
| `--top-k` | `16` | amostras herdadas por ciclo |
| `--model` | `deepseek-flash` | modelo da LLM (gerador) |
| `--learner-model` | `prajjwal1/bert-tiny` | BERT do learner (~4M params, rápido pra testar a arquitetura) |
| `--learner-epochs` | `3` | épocas de fine-tune do learner por ciclo |
| `--learner-batch-size` | `16` | batch do fine-tune do learner |
| `--lr` | `2e-5` | learning rate do learner |
| `--csv` | `dataset/pre-processed.csv` | caminho do Data 0 |
| `--device` | auto | `cuda`/`cpu` (vazio = auto) |

Os resultados por época são salvos em `outputs/results.json`.
