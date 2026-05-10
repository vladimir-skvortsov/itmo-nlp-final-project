"""Entry point for training a relation extraction model.

Usage:
    uv run scripts/train.py --config configs/typed_markers.yaml
    uv run scripts/train.py --config configs/baseline.yaml --seed 123
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# Ensure the src package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nerel_re.data.dataset import NO_RELATION, NERELDataset, load_splits
from nerel_re.data.negative_sampler import NegativeSampler
from nerel_re.data.tokenizer_utils import get_plain_marker_tokens, get_typed_marker_tokens
from nerel_re.training.trainer import RETrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def build_label_maps(examples_per_split):
    all_relations = {NO_RELATION}
    for examples in examples_per_split.values():
        all_relations.update(ex.relation for ex in examples)
    label2id = {lbl: i for i, lbl in enumerate(sorted(all_relations))}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a NEREL RE model.')
    parser.add_argument('--config', required=True, help='Path to YAML config file.')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg['training']['seed'] = args.seed

    data_dir = cfg['data']['data_dir']
    model_name = cfg['model']['name']
    use_typed = cfg['model'].get('use_typed_markers', True)

    logger.info('Loading NEREL splits from %s', data_dir)
    splits = load_splits(data_dir)

    logger.info('Running negative sampling')
    sampler = NegativeSampler(
        negative_ratio=cfg['data'].get('negative_ratio', 3.0),
        seed=cfg['training'].get('seed', 42),
    )
    splits['train'] = sampler.sample(splits['train'])

    label2id, id2label = build_label_maps(splits)
    entity_types = sorted(
        {ex.entity1.type for exs in splits.values() for ex in exs}
        | {ex.entity2.type for exs in splits.values() for ex in exs}
    )
    logger.info('%d relation types, %d entity types', len(label2id), len(entity_types))

    logger.info('Loading tokenizer: %s', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_tokens = get_typed_marker_tokens(entity_types) if use_typed else get_plain_marker_tokens()
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

    max_length = cfg['data'].get('max_length', 256)
    train_ds = NERELDataset(
        splits['train'], tokenizer, label2id, max_length, use_typed_markers=use_typed
    )
    dev_ds = NERELDataset(
        splits['dev'], tokenizer, label2id, max_length, use_typed_markers=use_typed
    )

    if use_typed:
        from nerel_re.models.typed_markers import TypedMarkersREModel

        model = TypedMarkersREModel(
            model_name=model_name,
            num_labels=len(label2id),
            entity_types=entity_types,
            tokenizer_vocab_size=len(tokenizer),
            dropout=cfg['model'].get('dropout', 0.1),
        )
    else:
        from nerel_re.models.baseline import BaselineREModel

        model = BaselineREModel(
            model_name=model_name,
            num_labels=len(label2id),
            dropout=cfg['model'].get('dropout', 0.1),
        )

    train_cfg = TrainingConfig(**cfg['training'])
    device = resolve_device()
    logger.info('Training on device: %s', device)

    trainer = RETrainer(
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
