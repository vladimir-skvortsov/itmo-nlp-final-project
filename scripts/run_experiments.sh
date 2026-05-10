#!/usr/bin/env bash
# Run all experiments sequentially and save results to output/.
# Usage: bash scripts/run_experiments.sh

set -euo pipefail

RESULTS_DIR="output/results"
mkdir -p "$RESULTS_DIR"

echo "=== [1/3] Baseline: RuBERT + plain markers ==="
uv run scripts/train.py --config configs/baseline.yaml
uv run scripts/evaluate.py \
    --config configs/baseline.yaml \
    --checkpoint output/baseline/best.pt \
    --split test \
    --per-class \
    | tee "$RESULTS_DIR/baseline.txt"

echo ""
echo "=== [2/3] Main: DeBERTa-v3 + typed markers ==="
uv run scripts/train.py --config configs/typed_markers.yaml
uv run scripts/evaluate.py \
    --config configs/typed_markers.yaml \
    --checkpoint output/typed_markers/best.pt \
    --split test \
    --per-class \
    | tee "$RESULTS_DIR/typed_markers.txt"

echo ""
echo "=== [3/3] LLM few-shot (Qwen2.5-7B) ==="
uv run scripts/predict.py \
    --config configs/llm.yaml \
    --split test \
    --output "$RESULTS_DIR/llm_predictions.json" \
    | tee "$RESULTS_DIR/llm.txt"

echo ""
echo "=== All experiments complete. Results saved to $RESULTS_DIR ==="
