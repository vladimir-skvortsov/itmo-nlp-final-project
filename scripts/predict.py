"""Run LLM few-shot inference on a split.

Usage:
    uv run scripts/predict.py --config configs/llm.yaml --split test
    uv run scripts/predict.py --config configs/llm.yaml --split test --n-few-shot 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nerel_re.data.dataset import NO_RELATION, load_splits
from nerel_re.data.negative_sampler import NegativeSampler
from nerel_re.evaluation.metrics import compute_metrics, print_results_table
from nerel_re.models.llm_prompt import LLMConfig, LLMRelationExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM few-shot RE inference.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test', choices=['dev', 'test'])
    parser.add_argument('--n-few-shot', type=int, default=None)
    parser.add_argument('--output', default=None, help='Save predictions to JSON file.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['data']['data_dir']
    logger.info('Loading NEREL splits')
    splits = load_splits(data_dir)
    sampler = NegativeSampler(negative_ratio=cfg['data'].get('negative_ratio', 3.0))
    splits['train'] = sampler.sample(splits['train'])

    all_relations = sorted({NO_RELATION} | {ex.relation for exs in splits.values() for ex in exs})
    label2id = {lbl: i for i, lbl in enumerate(all_relations)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    n_few_shot = args.n_few_shot or cfg['llm'].get('n_few_shot', 5)
    llm_cfg = LLMConfig(
        model_path=cfg['llm']['model_path'],
        allowed_relations=all_relations,
        n_few_shot=n_few_shot,
        max_new_tokens=cfg['llm'].get('max_new_tokens', 64),
        temperature=cfg['llm'].get('temperature', 0.0),
    )

    extractor = LLMRelationExtractor(
        config=llm_cfg,
        few_shot_examples=splits['train'][:200],
    )

    eval_examples = splits[args.split]
    logger.info('Running inference on %d examples', len(eval_examples))

    predictions = extractor.predict_batch(eval_examples)
    y_true = [label2id[ex.relation] for ex in eval_examples]
    y_pred = [label2id.get(p, label2id[NO_RELATION]) for p in predictions]

    metrics = compute_metrics(y_true, y_pred, id2label)
    print_results_table({'LLM few-shot': metrics})

    if args.output:
        out = [
            {'doc_id': ex.doc_id, 'true': ex.relation, 'pred': pred}
            for ex, pred in zip(eval_examples, predictions, strict=True)
        ]
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        logger.info('Predictions saved to %s', args.output)


if __name__ == '__main__':
    main()
