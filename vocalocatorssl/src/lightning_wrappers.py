import sys
import typing as tp

import lightning as L
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
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


class BufferCounterDict(torch.nn.Module):
    BUFFER_SIZE = 128
    persistent_buffer: torch.Tensor | None
    active_buffer: dict[int, int] | None

    def __init__(self, **kwargs):
        super().__init__()
        self.active_buffer = {}
        self.active_buffer.update(kwargs)
        self.register_buffer(
            "persistent_buffer",
            torch.zeros(BufferCounterDict.BUFFER_SIZE, dtype=torch.uint8),
            persistent=True,
        )
        self.is_frozen = False

    def load_buffer(self):
        if self.active_buffer:
            return
        if torch.all(self.persistent_buffer == 0):
            return

        self.active_buffer = utils.tensor_to_dict(self.persistent_buffer)

    def save_buffer(self):
        data = utils.dict_to_tensor(
            self.active_buffer, pad_to=BufferCounterDict.BUFFER_SIZE
        )
        self.persistent_buffer = data

    def __len__(self):
        self.load_buffer()
        return len(self.active_buffer)

    def __getitem__(self, key):
        key = str(key)
        self.load_buffer()
        if key not in self:
            if self.is_frozen:
                raise KeyError(f"Key {key} not found in buffer and buffer is frozen.")
            self[key] = len(self)
        return self.active_buffer[key]

    def __setitem__(self, key, value):
        self.load_buffer()
        if isinstance(value, torch.Tensor):
            value = int(value.item())
        elif not isinstance(value, int):
            value = int(value)

        key = str(key)
        self.active_buffer[key] = value
        self.save_buffer()

    def __contains__(self, key):
        key = str(key)
        return key in self.active_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps every element in x to the value stored in self[x]

        Args:
            x (torch.Tensor): Tensor of keys to look up

        Returns:
            torch.Tensor: Tensor of values corresponding to the keys
        """
        with torch.no_grad():
            output = torch.empty_like(x, dtype=torch.long)
            for key in torch.unique(x):
                output[x == key] = self[key.item()]
        return output


class AnimalIdentityEmbedding(torch.nn.Module):
    def __init__(self, num_animals: int, d_embedding: int):
        super().__init__()
        self.is_active = True
        animal_identity_embedding = torch.zeros(
            (num_animals, d_embedding),
            dtype=torch.float32,
        )
        torch.nn.init.xavier_uniform_(animal_identity_embedding)
        self.register_parameter(
            "animal_identity_embedding",
            torch.nn.Parameter(animal_identity_embedding, requires_grad=True),
        )
        # Helps map the animal ids in the dataset to a 0-indexed counter into the embedding
        self.animal_id_lookup = BufferCounterDict()

    def freeze(self):
        self.animal_id_lookup.is_frozen = True

    def eval(self):
        super().eval()
        self.freeze()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.animal_id_lookup.is_frozen = not mode
        return self

    def forward(
        self, location_embedding: torch.Tensor, animal_ids: torch.Tensor | None
    ) -> torch.Tensor:
        """Applies animal ID embeddings to location embeddings

        Args:
            location_embedding (torch.Tensor): Location embeddings (*batch, num_animals, d_embed)
            animal_ids (torch.Tensor | None): Animal IDs (*batch, num_animals)

        Returns:
            torch.Tensor: Location embeddings with animal identity added (*batch, num_animals, d_embed)
        """
        if not self.is_active or animal_ids is None:
            return location_embedding

        animal_id_indices = self.animal_id_lookup(animal_ids)
        animal_id_embs = torch.index_select(
            self.animal_identity_embedding, 0, animal_id_indices.view(-1)
        ).reshape(*animal_id_indices.shape, self.animal_identity_embedding.shape[1])
        return location_embedding + animal_id_embs

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True


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

        self.use_animal_identity: bool = config.get("use_animal_identity", True)
        num_animals_expected = self.config["dataloader"]["num_animals"]
        animal_id_embedding_dim = self.config["location_embedding_params"][
            "d_embedding"
        ]
        self.animal_id_embedding = AnimalIdentityEmbedding(
            num_animals=num_animals_expected, d_embedding=animal_id_embedding_dim
        )
        if not self.use_animal_identity:
            self.animal_id_embedding.is_active = False

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
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        animal_ids: torch.Tensor | None,
        shuffle: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes scores between audio and labels.

        Args:
            audio (torch.Tensor): Audio (batch, time, channels)
            labels (torch.Tensor): Labels (batch, 1+num_negative, animals, nodes, dims)
            animal_ids (torch.Tensor): Animal IDs (batch, 1+num_negative, animals)
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
            if animal_ids is not None:
                cur_negative_ids = animal_ids[torch.arange(bsz), positive_label_idx]
                animal_ids[torch.arange(bsz), positive_label_idx] = animal_ids[
                    torch.arange(bsz), 0
                ]
                animal_ids[torch.arange(bsz), 0] = cur_negative_ids
        else:
            positive_label_idx = torch.zeros(
                bsz, dtype=torch.long, device=labels.device
            )

        # audio_embeddings: (bsz, a_features)
        audio_embedding = self.audio_encoder(audio)
        # location_embeddings: (bsz, 1+num_negative, num_animals, l_features)
        location_embedding = self.location_encoder(labels)
        location_embedding = self.animal_id_embedding(location_embedding, animal_ids)

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
        audio, labels, animal_ids = batch["audio"], batch["labels"], batch["animal_ids"]
        audio = self.augmentation_transform(audio)

        scores, positive_scores, positive_label_index = self.forward(
            audio, labels, animal_ids
        )
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
        # Encourages the model to not score all animals associated with the same vocalization equally
        entropy_loss = score_entropy.mean() * self.entropy_coeff

        self.minibatch_idx += 1
        self.log(
            "loss_temperature",
            temperature,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "contrastive_loss_term",
            contrastive_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        self.log(
            "entropy_loss_term",
            entropy_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        self.log(
            "total_training_loss",
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
        audio, labels, animal_ids = batch["audio"], batch["labels"], batch["animal_ids"]

        # Shuffled to prevent weight explosion from inflating accuracy
        scores, positive_scores, positive_label_index = self.forward(
            audio, labels, animal_ids, shuffle=True
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
        animal_ids = batch.get("animal_ids", None)
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Create batch dim
        if len(labels.shape) == 4:
            labels = labels.unsqueeze(0)  # Create batch dim

        if not self.flags["predict_calibrate_mode"]:
            labels = labels.squeeze(1)  # Assume no negatives
            animal_ids = animal_ids.squeeze(1) if animal_ids is not None else None

        audio_embeddings = self.audio_encoder(audio)  # (b, feats)
        location_embeddings = self.location_encoder(labels)  # (b, n_animals, feats)
        if self.use_animal_identity:
            num_animals = self.animal_id_embedding.animal_identity_embedding.shape[0]
            if location_embeddings.shape[1] == num_animals and animal_ids is not None:
                # Make use of animal identity
                location_embeddings = self.animal_id_embedding(
                    location_embeddings, animal_ids
                )
            else:
                print(
                    f"Warning: location embeddings have {location_embeddings.shape[1]} animals but animal identity embedding has {num_animals} animals. Skipping animal identity embedding.",
                    file=sys.stderr,
                )
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
        ).transpose(
            1, 0, 2
        )  # (n_y, n_x, 3)
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
        ).transpose(
            2, 1, 0, 3
        )  # (n_z, n_y, n_x, 3)
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
                "monitor": "total_training_loss",
            },
        }


if __name__ == "__main__":
    print("Testing animal id")
    # Test animal id functionality
    num_animals = 2
    d_embedding = 128
    batch_size = 15
    num_negatives = 19

    fake_loc_embeddings = torch.randn(
        (batch_size, num_negatives + 1, num_animals, d_embedding)
    )
    animal_id_embedding = AnimalIdentityEmbedding(
        num_animals=num_animals, d_embedding=d_embedding
    )
    # Will allow us to see if it's being applied correctly
    animal_id_embedding.animal_identity_embedding.data[0, :] = float("nan")
    animal_id_embedding.animal_identity_embedding.data[1, :] = float("inf")

    animal_ids = torch.tensor([0, 1]).repeat(batch_size, num_negatives + 1, 1)
    print(
        "Before running forward: ", animal_id_embedding.animal_id_lookup.active_buffer
    )
    animal_id_embedding(fake_loc_embeddings, animal_ids)  # Seed the buffer
    print("After running forward: ", animal_id_embedding.animal_id_lookup.active_buffer)

    import tempfile

    # Test saving and reloading the embedding
    with tempfile.TemporaryDirectory() as tmpdir:
        print("testing save and load of animal id embedding")
        torch.save(animal_id_embedding.state_dict(), f"{tmpdir}/embedding.pt")
        new_embedding = AnimalIdentityEmbedding(num_animals=2, d_embedding=d_embedding)
        new_embedding.load_state_dict(torch.load(f"{tmpdir}/embedding.pt"))
        print("After load: ", new_embedding.animal_id_lookup.active_buffer)
        new_embedding(fake_loc_embeddings, animal_ids)
        print("After eval: ", new_embedding.animal_id_lookup.active_buffer)
        assert torch.isnan(new_embedding.animal_identity_embedding.data[0, :]).all()
        assert torch.isinf(new_embedding.animal_identity_embedding.data[1, :]).all()
