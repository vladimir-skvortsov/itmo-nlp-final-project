"""Train a NER model on the NEREL corpus.

Usage:
    python scripts/train_ner.py --config configs/ner_baseline.yaml
    python scripts/train_ner.py --config configs/ner_deberta.yaml --seed 123
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nerel_ner.data.dataset import (
    NERELNERDataset,
    build_label_maps,
    collect_entity_types,
    load_splits,
)
from nerel_ner.models.ner_model import NERModel
from nerel_ner.training.trainer import NERTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a NEREL NER model.')
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg['training']['seed'] = args.seed

    data_dir = cfg['data']['data_dir']
    model_name = cfg['model']['name']
    max_length = cfg['data'].get('max_length', 128)

    logger.info('Loading NEREL splits from %s', data_dir)
    splits = load_splits(data_dir)

    entity_types = collect_entity_types(splits)
    label2id, id2label = build_label_maps(entity_types)
    logger.info('%d entity types → %d BIO labels', len(entity_types), len(label2id))

    logger.info('Loading tokenizer: %s', model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = NERELNERDataset(splits['train'], tokenizer, label2id, max_length)
    dev_ds = NERELNERDataset(splits['dev'], tokenizer, label2id, max_length)
    logger.info('Train sentences: %d | Dev sentences: %d', len(train_ds), len(dev_ds))

    model = NERModel(
        model_name=model_name,
        num_labels=len(label2id),
        dropout=cfg['model'].get('dropout', 0.1),
    )

    train_cfg = TrainingConfig(**cfg['training'])
    device = resolve_device()
    logger.info('Training on device: %s', device)

    trainer = NERTrainer(
        model=model,
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        id2label=id2label,
        config=train_cfg,
        device=device,
    )
    best_f1 = trainer.train()
    logger.info('Training complete. Best dev macro F1: %.2f', best_f1)


if __name__ == '__main__':
    main()
