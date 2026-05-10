"""Training loop for RE models with W&B logging and checkpointing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
from nerel_re.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from torch import nn

    from nerel_re.data.dataset import NERELDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a single training run.

    Attributes:
        output_dir: Directory for checkpoints and logs.
        num_epochs: Total number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate for the encoder.
        head_learning_rate: Learning rate for the classification head
            (often higher than the encoder LR).
        warmup_ratio: Fraction of total steps used for LR warm-up.
        weight_decay: L2 regularisation coefficient.
        grad_clip: Maximum gradient norm (0.0 to disable).
        seed: Random seed for reproducibility.
        use_weighted_sampler: If ``True``, oversample under-represented classes.
        log_every_n_steps: W&B logging frequency.
        eval_every_n_epochs: How often to run evaluation on the dev set.
        wandb_project: W&B project name (``None`` disables W&B).
        run_name: W&B run name.
        extra: Additional metadata stored in the config (e.g. model name).
    """

    output_dir: str = 'output/runs'
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    use_weighted_sampler: bool = True
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    wandb_project: str | None = 'nerel-re'
    run_name: str = 'experiment'
    extra: dict[str, object] = field(default_factory=dict)


class RETrainer:
    """Manages the full training loop for a relation extraction model.

    Args:
        model: The PyTorch model to train.
        train_dataset: Training :class:`~nerel_re.data.dataset.NERELDataset`.
        dev_dataset: Validation dataset for early stopping / checkpointing.
        id2label: Mapping from integer class index to relation label string.
        config: :class:`TrainingConfig` with all hyperparameters.
        device: PyTorch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: NERELDataset,
        dev_dataset: NERELDataset,
        id2label: dict[int, str],
        config: TrainingConfig,
        device: str = 'cpu',
    ) -> None:
        """Initialise trainer, optimiser, scheduler, and W&B run."""
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.id2label = id2label
        self.config = config
        self.device = device
        self._best_f1 = 0.0
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)

        if config.wandb_project:
            wandb.init(project=config.wandb_project, name=config.run_name, config=vars(config))

    def train(self) -> float:
        """Run the full training loop.

        Returns:
            Best macro F1 observed on the dev set.
        """
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        dev_loader = self._make_loader(self.dev_dataset, shuffle=False)

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        optimizer = self._make_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss: torch.Tensor = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f'{loss.item():.4f}')

                if global_step % self.config.log_every_n_steps == 0 and self.config.wandb_project:
                    wandb.log({'train/loss': loss.item(), 'step': global_step})

            avg_loss = epoch_loss / len(train_loader)
            logger.info('Epoch %d — avg loss: %.4f', epoch, avg_loss)

            if epoch % self.config.eval_every_n_epochs == 0:
                metrics = self._evaluate(dev_loader)
                logger.info('Dev metrics: %s', metrics)
                if self.config.wandb_project:
                    wandb.log({f'dev/{k}': v for k, v in metrics.items()} | {'epoch': epoch})
                if metrics['macro_f1'] > self._best_f1:
                    self._best_f1 = metrics['macro_f1']
                    self._save_checkpoint('best.pt')
                    logger.info('New best F1: %.4f', self._best_f1)

        if self.config.wandb_project:
            wandb.finish()

        return self._best_f1

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:  # type: ignore[type-arg]
        """Run evaluation on a DataLoader and return metrics.

        Args:
            loader: DataLoader over the evaluation dataset.

        Returns:
            Dict with ``"macro_f1"``, ``"micro_f1"``, and ``"accuracy"``.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                outputs = self.model(**batch)
                preds = outputs['logits'].argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        return compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            id2label=self.id2label,
        )

    def _make_loader(self, dataset: NERELDataset, *, shuffle: bool) -> DataLoader:  # type: ignore[type-arg]
        """Create a DataLoader, optionally with weighted random sampling.

        Args:
            dataset: Source :class:`~nerel_re.data.dataset.NERELDataset`.
            shuffle: Whether to shuffle (ignored if weighted sampler is used).

        Returns:
            Configured :class:`torch.utils.data.DataLoader`.
        """
        sampler = None
        if shuffle and self.config.use_weighted_sampler:
            labels = list(dataset.iter_labels())
            class_counts = torch.bincount(torch.tensor(labels))
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            sample_weights = class_weights[torch.tensor(labels)]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset))
            shuffle = False

        # MPS does not support multiprocessing DataLoader; CUDA benefits from 2-4 workers
        device_str = str(self.device)
        num_workers = 0 if device_str.startswith('mps') else 2
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=device_str.startswith('cuda'),
        )

    def _make_optimizer(self) -> AdamW:
        """Build an AdamW optimiser with separate LRs for encoder and head.

        Returns:
            Configured :class:`torch.optim.AdamW` optimiser.
        """
        no_decay = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}
        encoder_params = [
            {
                'params': [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate,
            },
            {
                'params': [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate,
            },
        ]
        head_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'encoder' not in n],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.head_learning_rate,
            }
        ]
        return AdamW(encoder_params + head_params)

    def _save_checkpoint(self, filename: str) -> None:
        """Save model weights to ``output_dir/filename``.

        Args:
            filename: Checkpoint file name (e.g. ``"best.pt"``).
        """
        path = self._output_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info('Checkpoint saved to %s', path)
