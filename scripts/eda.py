import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nerel_re.data.dataset import load_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def plot_distribution(counts: Counter, title: str, output_path: Path, top_n: int = 30) -> None:
    most_common = counts.most_common(top_n)
    labels, values = zip(*most_common, strict=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.barh(labels, values, color='#4C72B0')
    ax.set_title(title)
    ax.set_xlabel('Count')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info('Saved %s', output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='EDA for NEREL.')
    parser.add_argument('--data-dir', required=True, help='Path to NEREL root directory.')
    parser.add_argument('--output-dir', default='output/eda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading NEREL from %s', args.data_dir)
    splits = load_splits(args.data_dir)

    stats: dict[str, object] = {}

    for split, examples in splits.items():
        rel_counts: Counter = Counter(ex.relation for ex in examples)
        e1_type_counts: Counter = Counter(ex.entity1.type for ex in examples)
        e2_type_counts: Counter = Counter(ex.entity2.type for ex in examples)
        entity_type_counts = e1_type_counts + e2_type_counts

        sent_lengths = [len(ex.sentence.split()) for ex in examples]

        stats[split] = {
            'n_examples': len(examples),
            'n_relation_types': len(rel_counts),
            'n_entity_types': len(entity_type_counts),
            'avg_sentence_words': round(sum(sent_lengths) / len(sent_lengths), 1),
            'most_common_relation': rel_counts.most_common(1)[0] if rel_counts else None,
        }

        print(f'\n=== {split.upper()} ===')
        print(f'  Examples   : {len(examples)}')
        print(f'  Relation types: {len(rel_counts)}')
        print('  Top-5 relations:')
        for rel, cnt in rel_counts.most_common(5):
            print(f'    {rel:<40} {cnt:>5}')

        plot_distribution(
            rel_counts,
            f'Relation distribution — {split}',
            output_dir / f'relation_dist_{split}.png',
        )
        plot_distribution(
            entity_type_counts,
            f'Entity type distribution — {split}',
            output_dir / f'entity_type_dist_{split}.png',
        )

        # Sentence length histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sent_lengths, bins=50, color='#55A868', edgecolor='white')
        ax.set_title(f'Sentence length (words) — {split}')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        fig.savefig(output_dir / f'sent_length_{split}.png', dpi=150)
        plt.close(fig)

    stats_path = output_dir / 'dataset_stats.json'
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    logger.info('Stats saved to %s', stats_path)


if __name__ == '__main__':
    main()
