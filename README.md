# Active-Inheritance Guided Synthetic Generation 🧬🤖

This repository implements the **Active Inheritance** framework for synthetic data generation in cybersecurity and fraudulent text detection tasks (e.g. phishing, fake news, fake ads).

Instead of generating synthetic data statically and passively (which often leads to model collapse), this pipeline uses an **Adversarial Evaluator** guided by a *learner* model (BERT). Only the synthetic samples that maximize the learner's uncertainty and error while keeping pool diversity are "inherited" into the next training cycle — acting as *deliberate practice*.

📄 **Full documentation:** [`docs/implementation.md`](docs/implementation.md)

---

## 🏗️ Pipeline Architecture

1. **Data 0 (real data):** the initial dataset (e.g. *Fake.br-Corpus*).
2. **Synthetic Generator:** LLMs (such as DeepSeek) instructed to generate new fraudulent examples.
3. **Learner (BERT):** a downstream classifier being trained to detect fraud.
4. **Adversarial Evaluator:** filters the LLM batch by computing the entropy (uncertainty) and error of the learner, plus the cosine distance (diversity via TF-IDF/embeddings).
5. **Active Inheritance:** the most challenging Top-K is selected, injected into the training set, and the cycle restarts.

---

## ⚙️ Installation & Setup

[uv](https://docs.astral.sh/uv/) recommended (Python 3.9+).

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/seu-usuario/active-inheritance-fraud.git
cd active-inheritance-fraud

# Create the venv (.venv) and install dependencies from pyproject.toml
uv sync

# Optional, no GPU: swap torch for the CPU-only build (much smaller)
uv remove torch && uv add torch --index https://download.pytorch.org/whl/cpu
```

For the LLM generator, copy `.env.example` to `.env` and fill it in:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
```

---

## 🚀 Usage

```bash
# Mock mode (offline, no LLM — perturbs real fake news)
uv run python main.py --generator mock --epochs 5 --top-k 16 --batch-size 64

# LLM mode (OpenAI-compatible endpoint)
uv run python main.py --generator openai --model deepseek-flash \
  --base-url https://api.deepseek.com --epochs 5 --top-k 16

# Core self-check (no torch)
uv run python tests/test_core.py
```

### Main arguments

| Argument | Default | Description |
|-----------|--------|-----------|
| `--generator` | `mock` | `mock` or `openai` |
| `--epochs` | `5` | Active Inheritance cycles (T) |
| `--batch-size` | `64` | synthetic batch size generated per cycle |
| `--top-k` | `16` | samples inherited per cycle |
| `--model` | `deepseek-flash` | LLM model (generator) |
| `--learner-model` | `prajjwal1/bert-tiny` | learner BERT (~4M params, fast architecture test) |
| `--learner-epochs` | `3` | learner fine-tune epochs per cycle |
| `--learner-batch-size` | `16` | learner fine-tune batch size |
| `--lr` | `2e-5` | learner learning rate |
| `--csv` | `dataset/pre-processed.csv` | Data 0 path |
| `--device` | auto | `cuda`/`cpu` (empty = auto) |

Per-epoch results are saved to `outputs/results.json`.
