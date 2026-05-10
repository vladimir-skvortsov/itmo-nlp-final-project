# Relation Extraction on NEREL

**ITMO University — NLP Course Final Project, Spring 2026**

Comparative study of relation extraction methods on the Russian [NEREL](https://github.com/nerel-ds/NEREL) corpus.
We compare a RuBERT baseline against DeBERTa-v3 with typed entity markers and LLM few-shot prompting.

## Task

Given a sentence and two annotated named entities, predict the semantic relation type between them
(or `no_relation`). We work with the **in-sentence** subset of NEREL's 49 relation types.

**Dataset:** NEREL — 933 Russian Wikinews documents, 56 K entities, 39 K relations.
**Metric:** Macro F1 (excluding `no_relation`), matching the original paper's evaluation protocol.
**Baseline to beat:** OpenNRE + RuBERT → F1 = 84.9 (Loukachevitch et al., RANLP 2021).

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Baseline** | OpenNRE-style classifier with `[CLS]` + `[E1]`/`[E2]` markers, RuBERT-base |
| 2 | **Typed Markers** | DeBERTa-v3-base + typed entity markers `[e1_PER]`…`[/e1_PER]` (Zhou et al., 2022) |
| 3 | **LLM few-shot** | Qwen2.5-7B (local via MLX) with structured few-shot prompt, no fine-tuning |

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment and install dependencies
uv sync

# 3. (Apple Silicon only) Install MLX for local LLM inference
uv sync --extra mlx

# 4. Install dev dependencies
uv sync --extra dev
```

## Data

Download NEREL and place under `data/nerel/`:

```bash
git clone https://github.com/nerel-ds/NEREL data/nerel
```

Then preprocess:

```bash
uv run scripts/eda.py --data-dir data/nerel --output-dir output/eda
```

## Training

All experiments are driven by YAML configs under `configs/`.

```bash
# Train baseline
uv run scripts/train.py --config configs/baseline.yaml

# Train main model (DeBERTa + typed markers)
uv run scripts/train.py --config configs/typed_markers.yaml

# Run all experiments sequentially
bash scripts/run_experiments.sh
```

## Evaluation

```bash
uv run scripts/evaluate.py \
    --config configs/typed_markers.yaml \
    --checkpoint output/typed_markers/best.pt
```

## LLM Few-shot

```bash
# Requires: uv sync --extra mlx
uv run scripts/predict.py \
    --config configs/llm.yaml \
    --split test
```

## Project Structure

```
nerel-re/
├── src/nerel_re/          # Main package
│   ├── data/              # Dataset loading, negative sampling, tokenization
│   ├── models/            # Model architectures
│   ├── training/          # Training loop, loss functions
│   └── evaluation/        # Metrics, reporting
├── scripts/               # CLI entry points (train, eval, predict, eda)
├── configs/               # YAML experiment configs
├── data/                  # Raw + processed data (see .gitignore)
├── output/                # Checkpoints, logs, plots (gitignored)
├── report/                # LaTeX report
└── tests/                 # Unit + integration tests
```

## Reproducibility

All experiments use fixed random seeds (set in config). Training was run on:
- **Hardware:** Apple M1 Max, 64 GB unified memory
- **Device:** `mps` (Metal Performance Shaders)
- **Python:** 3.11, PyTorch 2.3+

## References

- Loukachevitch et al. (2021). *NEREL: A Russian Dataset with Nested Named Entities, Relations and Events.* RANLP.
- Zhou et al. (2022). *An Improved Baseline for Sentence-level Relation Extraction.* AACL-IJCNLP.
- He et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.*
