from typing import Any

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy

from . import utils


class LVocalocator(L.LightningModule):
    def __init__(self, config: dict):
        """Lightning wrapper for Vocalocator model.

        Args:
            config (dict): Configuration dictionary. See utils.get_default_config for details.

        """
        super().__init__()
        self.config = config

        self.audio_encoder = utils.initialize_audio_embedder(config)
        self.location_encoder = utils.initialize_location_embedding(config)
        self.scorer = utils.initialize_scorer(config)
        self.augmentation_transform = utils.initialize_augmentations(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, audio, labels, shuffle: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes scores between audio and labels.

        Args:
            audio (_type_): Audio (batch, time, channels)
            labels (_type_): Labels (batch, 1+num_negative, mice, nodes, dims)
            shuffle (bool, optional): Whether labels are shuffled. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Scores and positive label indices
        """
        # Shuffle labels
        bsz = labels.shape[0]
        if shuffle:
            positive_label_idx = torch.randint(0, labels.shape[1], (bsz,)).to(
                labels.device
            )
            # Swap idx 0 and idx `positive_label_idx`
            cur_negatives = labels[torch.arange(bsz), positive_label_idx]
            labels[torch.arange(bsz), positive_label_idx] = labels[torch.arange(bsz), 0]
            labels[torch.arange(bsz), 0] = cur_negatives
        else:
            positive_label_idx = torch.zeros(
                bsz, dtype=torch.long, device=labels.device
            )

        # audio_embeddings: (bsz, features)
        audio_embedding = self.audio_encoder(audio)
        # location_embeddings: (bsz, 1+num_negative, features)
        location_embedding = self.location_encoder(labels)

        # Make audio embeddings broadcastable
        audio_embedding = audio_embedding.unsqueeze(1).expand_as(location_embedding)

        # Scores: (batch, 1+num_negative)
        scores = self.scorer(audio_embedding, location_embedding)

        return scores, positive_label_idx

    def training_step(self, batch: dict[str, torch.Tensor], *args: Any) -> torch.Tensor:
        """Executes a single iteration of the training loop and returns the
        (scalar) loss.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary with keys 'audio' and 'labels'

        Returns:
            torch.Tensor: Scalar loss value for this minibatch
        """
        audio, labels = batch["audio"], batch["labels"]
        audio = self.augmentation_transform(audio)

        scores, positive_label_index = self.forward(audio, labels)
        loss = self.loss_fn(scores, positive_label_index)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], *args: Any
    ) -> torch.Tensor:
        """Exectues a step of the validation pass and returns classification accuracy.
        Implementing this separately from the train_step to return accuracy.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary with keys 'audio' and 'labels'

        Returns:
            torch.Tensor: Scalar accuracy value for this minibatch
        """
        audio, labels = batch["audio"], batch["labels"]

        # Shuffled to prevent weight explosion from inflating accuracy
        scores, positive_label_index = self.forward(audio, labels, shuffle=True)
        # pred: (batch, )
        metric = MulticlassAccuracy(num_classes=scores.shape[1], average="micro").to(
            labels.device
        )
        acc = metric(scores, positive_label_index)

        self.log("val_acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        return acc

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler. This is a bit complicated
        becasue I want to use a cosine decay schedule with a warmup period.
        """
        total_steps = self.config["optimization"]["num_weight_updates"]
        warmup_steps = self.config["optimization"]["num_warmup_steps"]
        optimizer = utils.initialize_optimizer(self.config, self.parameters())
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
        chained_sched = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": chained_sched,
                "interval": "step",  # Call at every step, rather than epoch
                "frequency": 1,
            },
        }
