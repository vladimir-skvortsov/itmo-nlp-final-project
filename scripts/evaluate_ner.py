"""Evaluate a trained NER checkpoint on the test split.

Usage:
    python scripts/evaluate_ner.py --config configs/ner_baseline.yaml \\
        --checkpoint output/ner_baseline/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torch.utils.data import DataLoader

from nerel_ner.data.dataset import (
    LABEL_IGNORE_INDEX,
    NERELNERDataset,
    build_label_maps,
    collect_entity_types,
    load_splits,
)
from nerel_ner.evaluation.metrics import compute_metrics, per_class_report
from nerel_ner.models.ner_model import NERModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a NEREL NER checkpoint.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint.')
    parser.add_argument('--split', default='test', choices=['dev', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['data']['data_dir']
    model_name = cfg['model']['name']
    max_length = cfg['data'].get('max_length', 128)

    logger.info('Loading NEREL splits from %s', data_dir)
    splits = load_splits(data_dir)

    entity_types = collect_entity_types(splits)
    label2id, id2label = build_label_maps(entity_types)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = NERELNERDataset(splits[args.split], tokenizer, label2id, max_length)

    model = NERModel(
        model_name=model_name,
        num_labels=len(label2id),
        dropout=cfg['model'].get('dropout', 0.1),
    )
    device = resolve_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=cfg['training'].get('batch_size', 32), shuffle=False)

    all_preds: list[list[str]] = []
    all_refs: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop('labels')
            outputs = model(**batch)
            pred_ids = outputs['logits'].argmax(-1).cpu()

            for pred_row, label_row in zip(pred_ids, labels_batch.cpu(), strict=False):
                preds_str: list[str] = []
                refs_str: list[str] = []
                for p, t in zip(pred_row.tolist(), label_row.tolist(), strict=False):
                    if t == LABEL_IGNORE_INDEX:
                        continue
                    preds_str.append(id2label[p])
                    refs_str.append(id2label[t])
                all_preds.append(preds_str)
                all_refs.append(refs_str)

    metrics = compute_metrics(all_preds, all_refs)
    logger.info('%s metrics: %s', args.split.upper(), metrics)
    print(per_class_report(all_preds, all_refs))


if __name__ == '__main__':
    main()
