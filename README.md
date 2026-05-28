# Fine-grained NER on NEREL

**ITMO University — NLP Course Final Project, Spring 2026**

Named entity recognition on the Russian [NEREL](https://github.com/nerel-ds/NEREL) corpus.
We compare four transformer-based models for 29-class fine-grained NER in Russian news text.

## Task

Given a Russian sentence from the NEREL news wire corpus, predict a BIO tag sequence over
sub-word tokens covering 29 named entity types (Person, Organisation, Location, Country,
City, Disease, Law, Money, Work\_of\_Art, etc.).

**Dataset:** NEREL v1.1 — 907/101/100 train/dev/test documents, ~57 K entity mentions, 29 types.
**Metrics:** Span-level Macro F1 (primary) and Micro F1 via `seqeval`.

## Results

| Model | Macro F1 | Micro F1 | Split |
|---|---|---|---|
| ruBERT-base (baseline) | 66.83 | 80.50 | test |
| **XLM-RoBERTa-large (tuned)** | **69.38** | **80.59** | test |

*Best model: 256-token context, LR=7e-6, 8 epochs on A100.*

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install dev dependencies (for tests)
uv sync --extra dev
```

## Data

```bash
git clone https://github.com/nerel-ds/NEREL data/nerel
```

## Training

All experiments are driven by YAML configs under `configs/`.

```bash
# Baseline (ruBERT-base, ~10 min on A100)
python scripts/train_ner.py --config configs/ner_baseline.yaml

# Best model (XLM-RoBERTa-large tuned, ~65 min on A100)
python scripts/train_ner.py --config configs/ner_xlmr_tuned.yaml
```

## Evaluation

```bash
python scripts/evaluate_ner.py \
    --config configs/ner_xlmr_tuned.yaml \
    --checkpoint output/ner_xlmr_tuned/best.pt \
    --split test
```

## Tests

```bash
uv run python -m pytest tests/ -q
```

## Project Structure

```
itmo-nlp-final-project/
├── src/nerel_ner/          # Main package
│   ├── data/               # BRAT parser, BIO alignment, PyTorch Dataset
│   ├── models/             # NERModel wrapper (AutoModelForTokenClassification)
│   ├── training/           # Training loop, AdamW with layered LRs, W&B logging
│   └── evaluation/         # seqeval metrics, per-class report
├── scripts/                # CLI entry points (train_ner.py, evaluate_ner.py)
├── configs/                # YAML experiment configs
│   ├── ner_baseline.yaml   # ruBERT-base
│   ├── ner_deberta.yaml    # mDeBERTa-v3-base
│   ├── ner_xlmr.yaml       # XLM-RoBERTa-large (default)
│   └── ner_xlmr_tuned.yaml # XLM-RoBERTa-large (tuned, best)
├── report/                 # LaTeX report
└── tests/                  # Unit tests for data loading and metrics
```

## Reproducibility

All experiments use fixed random seeds (set in config). Training was run on:
- **Hardware:** Google Colab, NVIDIA A100 40 GB
- **Python:** 3.12, PyTorch 2.x, HuggingFace Transformers 4.x

## References

- Loukachevitch et al. (2021). *NEREL: A Russian Dataset with Nested Named Entities, Relations and Events.* RANLP 2021.
- Kuratov & Arkhipov (2019). *Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language.* arXiv:1905.07213.
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
