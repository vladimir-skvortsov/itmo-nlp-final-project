"""Training loop for NER models with W&B logging and checkpointing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
from nerel_ner.data.dataset import LABEL_IGNORE_INDEX
from nerel_ner.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from torch import nn

    from nerel_ner.data.dataset import NERELNERDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a training run.

    Attributes:
        output_dir: Directory to write checkpoints.
        num_epochs: Number of full passes over the training set.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for encoder parameters.
        head_learning_rate: Learning rate for the classification head.
        warmup_ratio: Fraction of total steps used for linear LR warm-up.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm (``<=0`` disables clipping).
        seed: Random seed.
        log_every_n_steps: W&B logging frequency.
        eval_every_n_epochs: How often to run dev evaluation.
        wandb_project: W&B project name (empty string disables W&B).
        run_name: W&B run name.
    """

    output_dir: str = 'output/ner'
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    head_learning_rate: float = 2e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    wandb_project: str = ''
    run_name: str = 'ner-run'
    # Internal — populated at runtime, not from YAML.
    label2id: dict[str, int] = field(default_factory=dict)
    id2label: dict[int, str] = field(default_factory=dict)


class NERTrainer:
    """Manages the full NER training loop.

    Args:
        model: The :class:`~nerel_ner.models.NERModel` to train.
        train_dataset: Training split dataset.
        dev_dataset: Development split dataset for evaluation.
        id2label: Index-to-label mapping.
        config: Training hyperparameters.
        device: PyTorch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: NERELNERDataset,
        dev_dataset: NERELNERDataset,
        id2label: dict[int, str],
        config: TrainingConfig,
        device: str = 'cpu',
    ) -> None:
        """Initialise trainer, move model to device, and start W&B if configured."""
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
            Best macro F1 observed on the dev set across all epochs.
        """
        device_str = str(self.device)
        num_workers = 0 if device_str.startswith('mps') else 2
        pin_memory = device_str.startswith('cuda')

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

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
        """Collect predictions and compute span-level NER metrics.

        Args:
            loader: DataLoader over the evaluation split.

        Returns:
            Metrics dict from :func:`~nerel_ner.evaluation.metrics.compute_metrics`.
        """
        self.model.eval()
        all_preds: list[list[str]] = []
        all_refs: list[list[str]] = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels_batch = batch.pop('labels')  # (batch, seq)
                outputs = self.model(**batch)
                pred_ids = outputs['logits'].argmax(-1).cpu()  # (batch, seq)

                for pred_row, label_row in zip(pred_ids, labels_batch.cpu(), strict=False):
                    preds_str: list[str] = []
                    refs_str: list[str] = []
                    for p, t in zip(pred_row.tolist(), label_row.tolist(), strict=False):
                        if t == LABEL_IGNORE_INDEX:
                            continue
                        preds_str.append(self.id2label[p])
                        refs_str.append(self.id2label[t])
                    all_preds.append(preds_str)
                    all_refs.append(refs_str)

        return compute_metrics(all_preds, all_refs)

    def _make_optimizer(self) -> AdamW:
        """Build AdamW with separate learning rates for encoder and head.

        Returns:
            Configured :class:`torch.optim.AdamW`.
        """
        no_decay = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}

        # Parameters belonging to the classification head (linear layer).
        head_params = {
            name
            for name, _ in self.model.named_parameters()
            if 'classifier' in name or 'qa_outputs' in name
        }

        param_groups = [
            {
                'params': [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in head_params and not any(nd in n for nd in no_decay)
                ],
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in head_params and any(nd in n for nd in no_decay)
                ],
                'lr': self.config.learning_rate,
                'weight_decay': 0.0,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters() if n in head_params
                ],
                'lr': self.config.head_learning_rate,
                'weight_decay': 0.0,
            },
        ]
        return AdamW(param_groups)

    def _save_checkpoint(self, filename: str) -> None:
        """Save model state dict to the output directory.

        Args:
            filename: File name within :attr:`_output_dir`.
        """
        path = self._output_dir / filename
        torch.save(self.model.state_dict(), path)
        logger.info('Checkpoint saved to %s', path)
