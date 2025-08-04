import typing as tp

import lightning as L
import numpy as np
import torch
from torch import optim
from torchmetrics.classification import MulticlassAccuracy

from . import utils


def info_nce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
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
        self.has_finetunified = False

        self.audio_encoder = utils.initialize_audio_embedder(config)
        self.location_encoder = utils.initialize_location_embedding(config)
        self.scorer = utils.initialize_scorer(config)
        self.augmentation_transform = utils.initialize_augmentations(config)
        self.flags = {
            "predict_calibrate_mode": False,
            "predict_gen_pmfs": False,
            "temperature_adjustment": 1.0,  # For calibration
        }
        self.entropy_coeff = config["optimization"].get("entropy_coeff", 1.0)

        self.register_buffer(
            "minibatch_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.cached_location_embeddings_grid = None

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
        linear_schedule = (
            self.config["optimization"].get("temperature_schedule", "exponential")
            == "linear"
        )
        cur_step = torch.clamp(self.minibatch_idx, min=0, max=num_steps)
        if linear_schedule:
            temp = initial_temp + ((final_temp - initial_temp) * cur_step / num_steps)
        else:
            temp = initial_temp * torch.exp(
                (cur_step / num_steps) * np.log(final_temp / initial_temp)
            )
        return temp

    def finetunify(self) -> None:
        if self.has_finetunified:
            return

        ft_config = self.config["finetune"]
        if "lora_rank" not in ft_config or "lora_alpha" not in ft_config:
            raise ValueError(
                "LoRA rank and alpha must be specified in the finetuning config"
            )
        # self.location_encoder.requires_grad_(False)
        # self.scorer.requires_grad_(False)
        if ft_config["method"] == "lora":
            print("Attempting to LoRAfy the model")
            self.audio_encoder.LoRAfy(
                lora_rank=self.config["finetune"]["lora_rank"],
                lora_alpha=self.config["finetune"]["lora_alpha"],
                lora_dropout=0.0,  # Not implemented yet
            )
        elif ft_config["method"] == "last_layers":
            print("Attempting to freeze all but the last layers of the model")
            num_layers = ft_config["num_last_layers"]
            self.audio_encoder.last_layer_finetunify(num_layers)
        self.has_finetunified = True

    def setup(self, stage) -> None:
        """Override to modify the model for finetuning if necessary."""
        if self.is_finetuning:
            self.finetunify()

    def forward(
        self, audio: torch.Tensor, labels: torch.Tensor, shuffle: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes scores between audio and labels.

        Args:
            audio (torch.Tensor): Audio (batch, time, channels)
            labels (torch.Tensor): Labels (batch, 1+num_negative, animals, nodes, dims)
            shuffle (bool, optional): Whether labels are shuffled. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]: Scores, difference
            between highest and second-highest positive scores, and positive label indices
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

        positive_scores = scores[
            torch.arange(bsz), positive_label_idx, :
        ]  # (batch, num_animals)
        return or_scores, positive_scores, positive_label_idx

    def training_step(
        self, batch: dict[str, torch.Tensor], *args: tp.Any
    ) -> torch.Tensor:
        """Executes a single iteration of the training loop and returns the
        (scalar) loss.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary with keys 'audio' and 'labels'

        Returns:
            torch.Tensor: Scalar loss value for this minibatch
        """
        temperature = self.compute_temperature()
        audio, labels = batch["audio"], batch["labels"]
        audio = self.augmentation_transform(audio)

        scores, positive_scores, positive_label_index = self.forward(audio, labels)
        positive_probs = torch.softmax(
            positive_scores / temperature, dim=-1
        )  # (batch, num_animals)
        positive_logprobs = torch.log(positive_probs + 1e-8)  # (batch, num_animals)
        score_entropy = -torch.sum(
            positive_probs * positive_logprobs, dim=-1
        )  # (batch,)
        contrastive_loss = info_nce(
            scores, positive_label_index, temperature=temperature
        )
        # Encourages the score spread to be supra-threshold
        entropy_loss = score_entropy.mean() * self.entropy_coeff

        self.minibatch_idx += 1
        self.log(
            "infonce_temperature",
            temperature,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "contrastive_loss",
            contrastive_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        self.log(
            "entropy_loss",
            entropy_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        self.log(
            "train_loss",
            contrastive_loss + entropy_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        return contrastive_loss + entropy_loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], *args: tp.Any
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
        scores, positive_scores, positive_label_index = self.forward(
            audio, labels, shuffle=True
        )
        # pred: (batch, )
        metric = MulticlassAccuracy(num_classes=scores.shape[1], average="micro").to(
            labels.device
        )
        acc = metric(scores, positive_label_index)

        positive_scores = torch.softmax(
            positive_scores / self.compute_temperature().item(), dim=-1
        )
        positive_scores = positive_scores.detach().cpu().numpy()
        confidence = positive_scores.max(axis=-1)  # (batch, )
        prop_assignable = (confidence > 0.95).mean().item()

        self.log("val_acc", acc, on_epoch=True, sync_dist=True, on_step=False)
        self.log(
            "uncalibrated_prop_assignable",
            prop_assignable,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            batch_size=len(audio),
        )
        return acc

    def on_predict_start(self):
        print(
            f"Starting prediction with temperature: {self.compute_temperature():.2f} and temperature adjustment: {self.flags['temperature_adjustment']:.2f}"
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], *args: tp.Any
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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

        if not self.flags["predict_calibrate_mode"]:
            labels = labels.squeeze(1)  # Assume no negatives

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        location_embeddings = self.location_encoder(labels)  # (b, n_animals, feats)
        audio_embeddings = audio_embeddings[:, None, :].expand(
            *location_embeddings.shape[:-1],
            -1,  # d_audio_embed not necessarily equal to d_loc_embed
        )

        scores = self.scorer(audio_embeddings, location_embeddings)  # (b, n_animals)
        # Ensure the same temperature used during training is applied at inference time
        temp_adjustment = self.flags["temperature_adjustment"]

        scores = scores / (self.compute_temperature() * temp_adjustment)

        if self.flags["predict_calibrate_mode"]:
            scores = torch.logsumexp(scores, dim=-1)  # sum over animals

        if self.flags["predict_gen_pmfs"]:
            # Generate PMFs for each animal
            pmfs = self.make_pmf(batch)
            return labels, scores, pmfs

        return labels, scores

    def make_pmf(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes score distributions for each candidate source location for each
        sound in the batch.

        Args:
            batch (dict[str, torch.Tensor]): Batch of audio samples and candidate locations.
                audio is expected to have shape (batch, time, channels)
                locations are expected to have shape (batch, num_negative + 1, num_animals, num_nodes, num_dims)

        Returns:
            torch.Tensor: Normalized probability distributions over arena (batch, num_theta, num_y, num_x)
        """
        is_3d = batch["labels"].shape[-1] == 3
        if is_3d:
            pmfs = self.make_pmf_3d(batch)
        else:
            pmfs = self.make_pmf_2d(batch)
        return pmfs

    def make_pmf_2d(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes score distributions for each candidate source location for each
        sound in the batch.

        Args:
            batch (dict[str, torch.Tensor]): Batch of audio samples and candidate locations.
                audio is expected to have shape (batch, time, channels)
                locations are expected to have shape (batch, num_negative + 1, num_animals, num_nodes, 2)

        Returns:
            torch.Tensor: Normalized probability distributions over arena (batch, num_theta, num_y, num_x)
        """
        audio = batch["audio"]
        labels = batch["labels"]
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Create batch dim
        if len(labels.shape) == 4:
            labels = labels.unsqueeze(0)  # Create batch dim

        labels = labels.squeeze(
            1
        )  # Assume no negatives  (b, n_animals, n_nodes, n_dims)

        head_to_nose_dist = torch.linalg.norm(
            labels[..., 0, :] - labels[..., 1, :], axis=-1
        ).mean()  # (,)
        # Grid resolution
        num_theta = 8
        num_xy = 55

        # Grid size
        theta_min = 0
        theta_max = 2 * np.pi
        arena_dims = self.config["dataloader"]["arena_dims"]
        aspect_ratio = arena_dims[0] / arena_dims[1]  # width / height
        x_min = -1
        x_max = 1
        y_min = -1 / aspect_ratio
        y_max = 1 / aspect_ratio

        theta = np.linspace(theta_min, theta_max, num_theta)
        animal_direction = (
            np.stack(
                [np.cos(theta), np.sin(theta)],
                axis=-1,
            )
            * head_to_nose_dist.cpu().item()
        )  # (n_theta, 2)
        animal_direction = torch.from_numpy(animal_direction).float().to(labels.device)

        head_location = np.stack(
            np.meshgrid(
                np.linspace(x_min, x_max, num_xy),
                np.linspace(y_min, y_max, num_xy),
                indexing="ij",
            ),  # returns tuple x,y with coords (x,y)
            axis=-1,
        ).transpose(1, 0, 2)  # (n_y, n_x, 3)
        head_location = torch.from_numpy(head_location).float().to(labels.device)

        # Get the nose location grid from the head locations and the animal directions
        head_location = head_location.reshape(1, num_xy, num_xy, 2)
        animal_direction = animal_direction.reshape(num_theta, 1, 1, 2)
        nose_location = head_location + animal_direction  # (n_theta, n_y, n_x, 3)

        head_location = head_location.expand_as(nose_location)
        animal_pose = torch.stack([nose_location, head_location], dim=-2)
        # (n_theta, n_y, n_x, 2, 3)

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        # No batch dimension in location_embeddings bc we only need one grid for all the animals
        # Compute location embeddings in batches to avoid OOM
        location_embeddings = self.cached_location_embeddings_grid
        if location_embeddings is None:
            try:
                location_embeddings = self.location_encoder(
                    animal_pose
                ).cpu()  # (n_theta, n_y, n_x, feats)
                location_embeddings = location_embeddings.reshape(
                    -1, location_embeddings.shape[-1]
                )
            except RuntimeError as e:
                print(f"OOM error in computing location embeddings: {e}")
                print("Attempting to compute in batches...")
                location_embeddings = torch.empty(
                    (
                        num_theta * num_xy * num_xy,
                        self.location_encoder.d_embedding,
                    ),
                    dtype=torch.float32,
                )
                flat_pose = animal_pose.reshape(num_theta * num_xy * num_xy, 2, 2)
                bsize = 2048
                num_batches = int(np.ceil(location_embeddings.shape[0] / bsize))
                for i in range(num_batches):
                    bstart = i * bsize
                    bend = min((i + 1) * bsize, location_embeddings.shape[0])
                    location_embeddings[bstart:bend] = self.location_encoder(
                        flat_pose[bstart:bend].cuda()
                    ).cpu()
            self.cached_location_embeddings_grid = location_embeddings

        pmfs = []
        temp_adjustment = self.flags["temperature_adjustment"]
        temp = self.compute_temperature().cpu() * temp_adjustment
        for audio_e in audio_embeddings:
            batch_size = 512
            audio_e = audio_e.unsqueeze(0).expand(batch_size, -1)
            scores = torch.zeros((num_theta * num_xy * num_xy,), dtype=torch.float64)
            for batch_start in range(0, len(scores), batch_size):
                batch_end = min(batch_start + batch_size, len(scores))
                loc_batch = location_embeddings[batch_start:batch_end].cuda()
                scores[batch_start:batch_end] = (
                    self.scorer(audio_e[: len(loc_batch)], loc_batch)
                    .cpu()
                    .to(torch.float64)
                    / temp
                )

            # This part gets expensive
            scores = torch.softmax(scores, dim=0)
            scores = scores.reshape(num_theta, num_xy, num_xy)
            pmfs.append(scores.to(torch.float32))
        return torch.stack(pmfs)

    def make_pmf_3d(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        labels = labels.squeeze(
            1
        )  # Assume no negatives  (b, n_animals, n_nodes, n_dims)

        head_to_nose_dist = torch.linalg.norm(
            labels[..., 0, :] - labels[..., 1, :], axis=-1
        ).mean()  # (,)
        # Grid resolution
        num_theta = 8
        num_phi = 4
        num_xy = 55
        num_z = 3

        arena_dims = self.config["dataloader"]["arena_dims"]
        aspect_ratio = arena_dims[0] / arena_dims[1]  # width / height

        # Grid size
        theta_min = 0
        theta_max = 2 * np.pi
        # phi_min = np.pi / 2 - 0.1
        phi_min = 0
        phi_max = np.pi / 2
        x_min = -1
        x_max = 1
        y_min = -1 / aspect_ratio
        y_max = 1 / aspect_ratio
        z_min = 0.05
        z_max = 0.2

        theta = np.linspace(theta_min, theta_max, num_theta).reshape(-1, 1)
        phi = np.linspace(phi_min, phi_max, num_phi).reshape(1, -1)
        animal_direction = (
            np.stack(
                [
                    np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.broadcast_to(np.cos(phi).reshape(1, -1), (num_theta, num_phi)),
                ],
                axis=-1,
            )
            * head_to_nose_dist.cpu().item()
        )  # (n_theta, n_phi, 3)
        animal_direction = torch.from_numpy(animal_direction).float().to(labels.device)

        head_location = np.stack(
            np.meshgrid(
                np.linspace(x_min, x_max, num_xy),
                np.linspace(y_min, y_max, num_xy),
                np.linspace(z_min, z_max, num_z),
                indexing="ij",
            ),  # returns tuple x,y,z with coords (x,y,z)
            axis=-1,
        ).transpose(2, 1, 0, 3)  # (n_z, n_y, n_x, 3)
        head_location = torch.from_numpy(head_location).float().to(labels.device)

        # Combine get the nose location from the head location and the animal direction
        head_location = head_location.reshape(1, 1, num_z, num_xy, num_xy, 3)
        animal_direction = animal_direction.reshape(num_theta, num_phi, 1, 1, 1, 3)
        nose_location = (
            head_location + animal_direction
        )  # (n_theta, n_phi, n_z, n_y, n_x, 3)

        head_location = head_location.expand_as(nose_location)
        animal_pose = torch.stack([nose_location, head_location], dim=-2)
        # (n_theta, n_phi, n_z, n_y, n_x, 2, 3)

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        # No batch dimension in location_embeddings bc we only need one grid for all the animals
        # Compute location embeddings in batches to avoid OOM
        location_embeddings = self.cached_location_embeddings_grid
        if location_embeddings is None:
            try:
                location_embeddings = self.location_encoder(
                    animal_pose.cuda()
                ).cpu()  # (n_theta, n_phi, n_z, n_y, n_x, feats)
                location_embeddings = location_embeddings.reshape(
                    -1, location_embeddings.shape[-1]
                )
            except RuntimeError as e:
                print(f"OOM error in computing location embeddings: {e}")
                print("Attempting to compute in batches...")
                location_embeddings = torch.empty(
                    (
                        num_theta * num_phi * num_z * num_xy * num_xy,
                        self.location_encoder.d_embedding,
                    ),
                    dtype=torch.float32,
                )
                flat_pose = animal_pose.reshape(
                    num_theta * num_phi * num_z * num_xy * num_xy, 2, 3
                )
                bsize = 2048
                num_batches = int(np.ceil(location_embeddings.shape[0] / bsize))
                for i in range(num_batches):
                    bstart = i * bsize
                    bend = min((i + 1) * bsize, location_embeddings.shape[0])
                    location_embeddings[bstart:bend] = self.location_encoder(
                        flat_pose[bstart:bend].cuda()
                    ).cpu()
            self.cached_location_embeddings_grid = location_embeddings

        pmfs = []
        temp = self.compute_temperature().cpu()
        for audio_e in audio_embeddings:
            batch_size = 512
            audio_e = audio_e.unsqueeze(0).expand(batch_size, -1)
            scores = torch.zeros(
                (num_theta * num_phi * num_z * num_xy * num_xy,), dtype=torch.float32
            )
            for batch_start in range(0, len(scores), batch_size):
                batch_end = min(batch_start + batch_size, len(scores))
                loc_batch = location_embeddings[batch_start:batch_end].cuda()
                scores[batch_start:batch_end] = (
                    self.scorer(audio_e[: len(loc_batch)], loc_batch).cpu()
                    # .to(torch.float64)
                    / temp
                )

            # This part gets expensive
            scores = torch.softmax(scores, dim=0)
            # Sum over phi, z
            scores = scores.reshape(num_theta, num_phi, num_z, num_xy, num_xy)
            scores = scores.sum(dim=(1, 2))  # (n_theta, n_y, n_x)
            pmfs.append(scores)
        return torch.stack(pmfs)

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for a simple reduce-on-plateau
        setup."""
        optimizer = utils.initialize_optimizer(
            self.config,
            filter(lambda p: p.requires_grad, self.parameters()),
            is_finetuning=self.is_finetuning,
        )
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=50, factor=0.5, mode="min"
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
