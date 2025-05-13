from typing import Any

import lightning as L
import numpy as np
import torch
from torch import optim
from torchmetrics.classification import MulticlassAccuracy

from . import utils


def info_nce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
):
    logits = logits / temperature
    return torch.logsumexp(logits, dim=1).mean(dim=0) - logits.gather(
        dim=1, index=labels.unsqueeze(1)
    ).mean(dim=0)


class LVocalocator(L.LightningModule):
    def __init__(self, config: dict, is_finetuning: bool = False):
        """Lightning wrapper for Vocalocator model.

        Args:
            config (dict): Configuration dictionary. See utils.get_default_config for details.

        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.is_finetuning = is_finetuning

        self.audio_encoder = utils.initialize_audio_embedder(config)
        self.location_encoder = utils.initialize_location_embedding(config)
        self.scorer = utils.initialize_scorer(config)
        self.augmentation_transform = utils.initialize_augmentations(config)
        self.test_flag = False  # Used during predict_step to determine behavior

        self.register_buffer(
            "minibatch_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def compute_temperature(self) -> torch.Tensor:
        """Computes the temperature for the current traninig step. This controls how strongly hard
        negative samples are penalized by the loss function.

        Args:
            batch_idx (int): Number of minibatches processed so far.

        Returns:
            float: Temperature for the current training step.
        """
        num_steps, initial_temp, final_temp = (
            self.config["optimization"]["num_temperature_steps"],
            self.config["optimization"]["initial_temperature"],
            self.config["optimization"]["final_temperature"],
        )
        cur_step = torch.clamp(self.minibatch_idx, min=0, max=num_steps)
        temp = initial_temp + ((final_temp - initial_temp) * cur_step / num_steps)
        return temp

    def on_train_start(self) -> None:
        """Override to modify the model for finetuning if necessary."""
        if self.is_finetuning:
            ft_config = self.config["finetune"]
            if "lora_rank" not in ft_config or "lora_alpha" not in ft_config:
                raise ValueError(
                    "LoRA rank and alpha must be specified in the finetuning config"
                )
            if ft_config["method"] == "lora":
                print("Attempting to LoRAfy the model")
                self.audio_encoder.LoRAfy(
                    lora_rank=self.config["finetune"]["lora_rank"],
                    lora_alpha=self.config["finetune"]["lora_alpha"],
                    lora_dropout=0.0,  # Not implemented yet
                )
            print(self)

    def forward(
        self, audio: torch.Tensor, labels: torch.Tensor, shuffle: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes scores between audio and labels.

        Args:
            audio (torch.Tensor): Audio (batch, time, channels)
            labels (torch.Tensor): Labels (batch, 1+num_negative, animals, nodes, dims)
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

        # audio_embeddings: (bsz, a_features)
        audio_embedding = self.audio_encoder(audio)
        # location_embeddings: (bsz, 1+num_negative, num_animals, l_features)
        location_embedding = self.location_encoder(labels)

        # Make audio embeddings broadcastable
        audio_embedding = audio_embedding[:, None, None, :].expand(
            *location_embedding.shape[:-1], -1
        )

        # Scores: (batch, 1+num_negative, num_animals)
        scores = self.scorer(audio_embedding, location_embedding)

        # proportional to log prob of animal A or animal B or ...
        or_scores = torch.logsumexp(scores, dim=-1)
        # Shape: (batch, 1 + num_negative)

        return or_scores, positive_label_idx

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
        temperature = self.compute_temperature()
        loss = info_nce(scores, positive_label_index, temperature=temperature)

        self.minibatch_idx += 1
        self.log("infonce_temperature", temperature, on_step=True, sync_dist=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=False)
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

        self.log("val_acc", acc, on_epoch=True, sync_dist=True, on_step=False)
        return acc

    def on_predict_start(self):
        print(f"Starting prediction with temperature: {self.compute_temperature():.2f}")

    def predict_step(
        self, batch: dict[str, torch.Tensor], *args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes score distributions for each candidate source location for each
        sound in the batch.

        Args:
            batch (dict[str, torch.Tensor]): Batch of audio samples and candidate locations.
                audio is expected to have shape (batch, time, channels)
                locations are expected to have shape (batch, num_negative + 1, num_animals, num_nodes, num_dims)

        Returns:
            torch.Tensor: Labels provided as input
            torch.Tensor: Scores for each animal (batch, n_animals)
        """
        audio = batch["audio"]
        labels = batch["labels"]
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Create batch dim
        if len(labels.shape) == 4:
            labels = labels.unsqueeze(0)  # Create batch dim

        if self.test_flag:
            labels = labels.squeeze(2)  # Assume only one animal per label
        else:
            labels = labels.squeeze(1)  # Assume no negatives

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        location_embeddings = self.location_encoder(labels)  # (b, n_animals, feats)
        audio_embeddings = audio_embeddings[:, None, :].expand(
            *location_embeddings.shape[:-1],
            -1,  # d_audio_embed not necessarily equal to d_loc_embed
        )

        scores = self.scorer(audio_embeddings, location_embeddings)  # (b, n_animals)
        # Ensure the same temperature used during training is applied at inference time
        scores = scores / self.compute_temperature()

        return labels, scores

    def make_pmf(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes score distributions for each candidate source location for each
        sound in the batch.

        Args:
            batch (dict[str, torch.Tensor]): Batch of audio samples and candidate locations.
                audio is expected to have shape (batch, time, channels)
                locations are expected to have shape (batch, num_negative + 1, num_animals, num_nodes, num_dims)

        Returns:
            torch.Tensor: Normalized probability distributions over arena (batch, num_angle, num_y, num_x)
        """
        audio = batch["audio"]
        labels = batch["labels"]
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Create batch dim
        if len(labels.shape) == 4:
            labels = labels.unsqueeze(0)  # Create batch dim

        labels_to_save = labels[:, 0, :, :, :]  # Assume no negatives
        labels = labels_to_save[:, 0, :, :]  # (b, n_nodes, n_dims)

        head_to_nose = labels[:, 0, :] - labels[:, 1, :]  # (b, 3)
        # Expand to multiple orientations
        # We will consider shifting the orientation 15, 30, 45 in either direction
        num_angles = 120

        # Distance from head to nose ignoring z dimension
        h_t_n_xy_mag = torch.linalg.norm(head_to_nose[:, :2], dim=-1)  # (b, )
        # Yaw angle from head to nose, relative to x+
        new_h_t_n_xy_angle = (
            torch.from_numpy(np.linspace(0, 2 * np.pi, num_angles))
            .to(labels.device)
            .to(labels.dtype)
        ).expand(labels.shape[0], -1)  # (b, n_angle)
        new_h_t_n = torch.stack(
            [
                h_t_n_xy_mag[:, None] * torch.cos(new_h_t_n_xy_angle),  # (b, n_angle)
                h_t_n_xy_mag[:, None] * torch.sin(new_h_t_n_xy_angle),  # (b, n_angle)
                head_to_nose[:, 2].unsqueeze(1).expand(-1, num_angles),  # (b, n_angle)
            ],
            dim=-1,
        )  # (b, n_angle, 3)

        grid = np.stack(
            np.meshgrid(
                np.linspace(-1.1, 1.1, 110),  # x
                np.linspace(-1.1, 1.1, 110),  # y
                [labels[:, 0, 2].mean(dim=0).cpu().numpy()],  # z
            ),
            axis=-1,
        ).squeeze()  # (n_y, n_x, 3)
        grid_head = (
            torch.from_numpy(grid).float().to(labels.device)
        )  # (n_z, n_y, n_x, 3)
        grid_head = grid_head.reshape(1, 1, -1, 3)  # (1 (b), 1(n_angle), n_grid, 3)
        # create nose points
        grid_nose = grid_head + new_h_t_n[:, :, None, :]  # (b, n_angle, n_grid, 3)
        grid_head = grid_head.expand_as(grid_nose)  # (b, n_grid, 3)
        grid = torch.stack([grid_nose, grid_head], dim=-2)  # (b, n_angle, n_grid, 2, 3)

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        location_embeddings = self.location_encoder(
            grid
        )  # (b, n_angle, n_grid, d_embed)

        audio_embeddings = audio_embeddings[:, None, None, :].expand(
            *location_embeddings.shape[:-1],
            -1,  # d_audio_embed not necessarily equal to d_loc_embed
        )
        scores = self.scorer(
            audio_embeddings, location_embeddings
        )  # (b, n_angle, n_grid)
        scores = scores.reshape(-1, num_angles, 110, 110)  # (b*n_angle, n_grid)
        scores = scores / self.compute_temperature()
        return scores.to(torch.float16)

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for a simple reduce-on-plateau
        setup."""
        optimizer = utils.initialize_optimizer(self.config, self.parameters())
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=20, factor=0.5, mode="min"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }
