import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nerel_re.data.dataset import NO_RELATION, NERELDataset, load_splits
from nerel_re.data.negative_sampler import NegativeSampler
from nerel_re.data.tokenizer_utils import get_plain_marker_tokens, get_typed_marker_tokens
from nerel_re.evaluation.metrics import compute_metrics, per_class_report, print_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RE checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--per-class", action="store_true", help="Print per-class F1 breakdown.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    use_typed = cfg["model"].get("use_typed_markers", True)
    data_dir = cfg["data"]["data_dir"]
    max_length = cfg["data"].get("max_length", 256)

    logger.info("Loading NEREL splits")
    splits = load_splits(data_dir)
    sampler = NegativeSampler(negative_ratio=cfg["data"].get("negative_ratio", 3.0))
    splits["train"] = sampler.sample(splits["train"])

    all_relations = {NO_RELATION}
    for examples in splits.values():
        all_relations.update(ex.relation for ex in examples)
    label2id = {lbl: i for i, lbl in enumerate(sorted(all_relations))}
    id2label = {i: lbl for lbl, i in label2id.items()}

    entity_types = sorted(
        {ex.entity1.type for exs in splits.values() for ex in exs}
        | {ex.entity2.type for exs in splits.values() for ex in exs}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_tokens = get_typed_marker_tokens(entity_types) if use_typed else get_plain_marker_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    eval_ds = NERELDataset(
        splits[args.split], tokenizer, label2id, max_length, use_typed_markers=use_typed
    )

    if use_typed:
        from nerel_re.models.typed_markers import TypedMarkersREModel

        model = TypedMarkersREModel(
            model_name=model_name,
            num_labels=len(label2id),
            entity_types=entity_types,
            tokenizer_vocab_size=len(tokenizer),
        )
    else:
        from nerel_re.models.baseline import BaselineREModel

        model = BaselineREModel(model_name=model_name, num_labels=len(label2id))

    device = resolve_device()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    from torch.utils.data import DataLoader

    loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch)["logits"]
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, id2label)
    print_results_table({cfg.get("run_name", model_name): metrics})

    if args.per_class:
        print("\n" + per_class_report(all_labels, all_preds, id2label))


if __name__ == "__main__":
    main()
