"""Relation Extraction on the Russian NEREL corpus.

Methods compared:
- RuBERT baseline with standard entity markers
- DeBERTa-v3 with typed entity markers (main contribution)
- LLM few-shot prompting (Qwen2.5-7B via MLX)
"""

__version__ = '0.1.0'
