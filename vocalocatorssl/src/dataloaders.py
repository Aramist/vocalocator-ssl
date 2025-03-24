"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.stats.distributions import beta

# from scipy.spatial import KDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class VocalizationDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        *,
        arena_dims: Optional[Union[np.ndarray, list[float]]] = None,
        nodes: Optional[list[str]] = None,
        crop_length: int,
        crop_randomly: bool = False,
        num_negative_samples: int = 1,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
        inference: bool = False,
        difficulty_range: Optional[tuple[float, float]] = None,
        num_difficulty_steps: int = 80_000,
    ):
        """
        Args:
            dataset_path (Path): Path to HDF5 dataset
            arena_dims (Optional[Union[np.ndarray, Tuple[float, float]]], optional): Dimensions of the arena in mm. Used to scale labels.
            nodes (Optional[list[str]], optional): List of skeleton nodes to retrieve from the dataset. Defaults to None, which will retrieve only the first node.
            crop_length (int): Length of audio samples to return.
            crop_randomly (bool, optional): When false, data will be cropped starting at the first sample. Defaults to False.
            num_negative_samples (int, optional): Number of negative samples to retrieve for each positive example. Defaults to 1.
            index (Optional[np.ndarray], optional): An array of indices to use for this dataset. Defaults to None, which will use the full dataset
            normalize_data (bool, optional): Whether to normalize the audio data to have zero mean and unit variance. Defaults to True.
            inference (bool, optional): Whether this dataset is for inference. Defaults to False
        """
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        self.datapath = dataset_path
        dataset = h5py.File(self.datapath, "r")
        # dataset cannot exist as a member of the object until after pytorch has cloned and
        # spread the dataset objects across multiple processes.
        # This is because h5py handles cannot be pickled and pytorch uses pickle under the hood
        # I get around this by re-initializing the h5py.File lazily in __getitem__
        self.dataset: Optional[h5py.File] = None

        if not isinstance(arena_dims, np.ndarray):
            arena_dims = np.array(arena_dims).astype(np.float32)

        if "length_idx" not in dataset or "audio" not in dataset:
            raise ValueError("Improperly formatted dataset")

        self.crop_randomly: bool = crop_randomly
        self.arena_dims: np.ndarray = arena_dims
        self.crop_length: int = crop_length
        self.num_negative_samples: int = num_negative_samples
        self.index: np.ndarray = (
            index if index is not None else np.arange(len(dataset["length_idx"]) - 1)
        )
        self.normalize_data: bool = normalize_data
        self.inference: bool = inference
        self.length: int = len(self.index)
        self.inverse_index = {v: k for k, v in enumerate(self.index)}

        self.difficulty_range: Optional[tuple[float, float]] = difficulty_range
        self.num_difficulty_steps: int = num_difficulty_steps
        self.cur_difficulty_step: int = 0
        # Array which stores the difficulty of each vocalization
        self.vocalization_difficulties: Optional[np.ndarray] = None
        # Map from indices in the sorted difficulty array to the indices in the dataset
        self.difficulty_sorting: Optional[np.ndarray] = None
        # Maps from indices in the dataset to indices in the difficulty sorting
        self.inv_difficulty_sorting: Optional[np.ndarray] = None

        # Determine which nodes indices to select
        if nodes is None:
            self.node_indices = np.array([0])
        else:
            if "node_names" not in dataset:
                raise ValueError("Dataset does not contain node names")
            dset_node_names = list(map(bytes.decode, dataset["node_names"]))
            self.node_indices = [
                i for i, node in enumerate(dset_node_names) if node in nodes
            ]

            if not self.node_indices:
                raise ValueError(
                    "No nodes found in dataset with the given names: {}".format(nodes)
                )
            self.node_indices = np.array(self.node_indices)

        # Ensure parallel workers don't produce the same stream of vocalizations
        worker_info = torch.utils.data.get_worker_info()
        seed = 0 if worker_info is None else worker_info.seed
        self.rng = np.random.default_rng(seed)

        if self.difficulty_range is not None:
            # Expected shape: (dataset_size, n_animals, n_nodes, n_dims)
            animal_configurations = dataset["locations"][:, :, self.node_indices[0], :]
            animal_configurations = animal_configurations[self.index, ...]
            self.vocalization_difficulties = self.__compute_difficulties(
                animal_configurations
            )
            # Points with a low scalar value for `difficulty` are more difficult
            self.difficulty_sorting = np.argsort(self.vocalization_difficulties)
            self.inv_difficulty_sorting = np.argsort(self.difficulty_sorting)

        dataset.close()  # The dataset cannot be pickled by pytorch

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wraps the __processed_data_for_index__ method to provide lazy loading
        of the h5py.File object and abstraction of the dataset index.

        Args:
            idx (int): Index of the vocalization

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The audio and locations (+ and -) for
        the index. Audio is of shape (n_samples, n_channels).
        Locations are of shape (n_negative + 1, n_animals, n_nodes, n_dims)
        """
        if self.dataset is None:
            # Lazily recreate the dataset object in the child process
            self.dataset = h5py.File(self.datapath, "r")
        true_idx = self.index[idx]
        if self.inference:
            return self.__inference_data_for_index__(true_idx)
        return self.__processed_data_for_index__(true_idx)

    def __make_crop(self, audio: torch.Tensor, crop_length: int) -> torch.Tensor:
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        audio_len, _ = audio.shape
        valid_range = audio_len - crop_length
        if valid_range <= 0:  # Audio is shorter than desired crop length, pad right
            pad_size = crop_length - audio_len
            # will fail if input is numpy array
            return F.pad(audio, (0, 0, 0, pad_size))
        if self.crop_randomly:
            range_start = 0  # Ensure the output is deterministic at inference time
        else:
            range_start = self.rng.integers(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def sample_negative_location(self, idx: int) -> torch.Tensor:
        """Samples a negative ground truth from the dataset. This is defined as a set of animal
        positions from a different time point than the positive sample.

        Args:
            idx (int): True index of the positive sample in the dataset

        Returns:
            torch.Tensor: The negative sample. Shape: (n_animals, n_node, n_dim)
        """
        if self.difficulty_range is not None:
            return self.sample_negatives_with_difficulty(idx)
        choices = np.delete(self.index, self.inverse_index[idx])
        neg_idx = self.rng.choice(choices)
        return self.__label_for_index(neg_idx)

    def __audio_for_index(self, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = self.dataset["length_idx"][idx : idx + 2]
        audio = self.dataset["audio"][start:end, ...]
        return torch.from_numpy(audio.astype(np.float32))

    def __label_for_index(self, idx: int) -> torch.Tensor:
        """Gets the ground truth source location for the vocalization at the given index

        Args:
            idx (int): Index of the vocalization

        Returns:
            torch.Tensor: The source location of the vocalization, if available. Shape: (n_animals, n_node, n_dim). Unit: dataset unit
        """

        locs = self.dataset["locations"][idx, ..., self.node_indices, :]
        # either (n_animals, n_nodes, n_dims) if len(locs.shape) == 3
        # or (n_nodes, n_dims)
        if len(locs.shape) == 2:
            locs = locs[None, ...]
        locs = torch.from_numpy(locs.astype(np.float32))
        return locs

    def sample_negative_location(self, idx: int) -> torch.Tensor:
        """Samples a negative ground truth from the dataset. This is defined as a set of animal
        positions from a different time point than the positive sample.

        Args:
            idx (int): True index of the positive sample in the dataset

        Returns:
            torch.Tensor: The negative sample. Shape: (n_animals, n_node, n_dim). Unit: dataset unit
        """

        choices = np.delete(self.index, self.inverse_index[idx])
        neg_idx = self.rng.choice(choices)
        return self.__label_for_index(neg_idx)

    def sample_rand_locations(self, num: int) -> torch.Tensor:
        """Samples random locations from the dataset. These are used for performing inference
        using the trained models by comparing the scores of fake locations to both of the given
        locations.

        Args:
            num (int): Number of random locations to sample

        Returns:
            torch.Tensor: Batch of scaled, random locations. Shape: (num, n_node, n_dim). Unit: arb.
        """
        if self.dataset is None:
            # Lazily recreate the dataset object in the child process
            self.dataset = h5py.File(self.datapath, "r")

        indices = self.rng.choice(len(self.dataset["locations"]), num, replace=True)
        labels = torch.stack([self.__label_for_index(idx) for idx in indices])
        # Labels shape: (num, n_animals, n_nodes, n_dims)
        # May have more than one animal in each location, randomly choose one per location
        animal_choices = torch.from_numpy(
            self.rng.integers(0, labels.shape[1], size=(num,))
        )
        labels = labels[torch.arange(num), animal_choices, ...]
        return self.scale_labels(labels)  # Shape: (num, n_nodes, n_dims)

    @property
    def difficulty(self) -> Optional[float]:
        """Returns the current difficulty level, which is a value between 0 and 1."""
        if self.difficulty_range is None:
            return None

        return self.cur_difficulty_step / self.num_difficulty_steps

    def step_difficulty(self):
        self.cur_difficulty_step = min(
            self.cur_difficulty_step + 1, self.num_difficulty_steps
        )

    def __compute_difficulties(self, animal_configurations: np.ndarray) -> np.ndarray:
        """Computes the difficulty of a batch of animal positions based on the distance
        between the animals. Lower values for difficulty indicate more difficult samples,
        as the animals have less distance between them.

        Args:
            animal_configurations (np.ndarray): Array of animal positions. Shape: (n_samples, n_animals, n_nodes, n_dims)

        Returns:
            np.ndarray: Difficulties of shape (n_samples,)
        """

        n_samples, n_animals, _, _ = animal_configurations.shape
        animal_configurations = animal_configurations.reshape(n_samples, n_animals, -1)
        # How spread out are the animals?
        spread = animal_configurations.std(axis=1)  # Shape: (n_samples, n_dims)
        # Combine dims. For the two-animal case I think this reduces to the euclidean distance
        difficulty = np.linalg.norm(spread, axis=1)
        return difficulty

    def sample_negatives_with_difficulty(self, idx: int) -> torch.Tensor:
        """Samples a negative frame from the dataset for index `idx` based on the
        current difficulty level. This is done by computing a distribution over
        the dataset based on each frame's difficulty and sampling from it to obtain
        an index.

        Args:
            idx (int): Current index. This function guarantees the sampled index will be distinct

        Returns:
            torch.Tensor: Animal poses for the negative frame. Shape: (n_animals, n_nodes, n_dims)
        """
        min_d, max_d = self.difficulty_range
        cur_difficulty = (
            self.difficulty * (max_d - min_d) + min_d
        )  # float between min and max difficulty
        beta = 10**cur_difficulty
        alpha = (
            1 / beta
        )  # Smaller values of cur_difficulty yield samples of the beta distributino
        # closer to zero, which corresponds to early (low spread) indices in the sorting
        # index
        sample = self.rng.beta(alpha, beta, 1).item()  # Sampled value between 0 and 1
        # Like the input `idx`, this is a value between 0 and len(self)
        # that is, an index into `self.index` rather than the underlying dataset
        sampled_idx = int(
            sample * (len(self) - 1)
        )  # in clopen interval [0, len(self) )
        # Ensure that `idx` isn't sampled
        if sampled_idx == self.inv_difficulty_sorting[idx]:
            sampled_idx += 1 if sampled_idx < len(self) - 1 else -1
        # Map the sampled index back to the original dataset
        idx_to_return = self.difficulty_sorting[sampled_idx]

    def scale_audio(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Scales the inputs to have zero mean and unit variance."""

        if self.normalize_data:
            audio = (audio - audio.mean()) / audio.std()

        return audio

    def scale_labels(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Scales the labels from millimeter units to an arbitrary unit with range [-1, 1]."""

        # Shift range to [-1, 1], but keep units same across dimensions
        # Origin remains at center of arena
        scale_factor = self.arena_dims.max() / 2
        labels = labels / scale_factor

        return labels

    def __processed_data_for_index__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the audio and locations from a given index in the dataset. This function
        should not be called directly.

        Args:
            idx (int): Index of the vocalization

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The audio and locations (+ and -) for
        the index. Audio is of shape (n_samples, n_channels).
        Locations are of shape (n_negative + 1, n_animals, n_nodes, n_dims)
        """
        sound = self.__audio_for_index(idx)
        sound = self.__make_crop(sound, self.crop_length)
        positive_location = self.__label_for_index(idx)
        negative_locations = [
            self.sample_negative_location(idx) for _ in range(self.num_negative_samples)
        ]

        locations = torch.stack([positive_location] + negative_locations, dim=0)
        sound, locations = self.scale_audio(sound), self.scale_labels(locations)

        return sound, locations

    def __inference_data_for_index__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the audio and locations for inference. This differs from standard procedure in that
        for each true location, a set of random locations are also returned.

        Args:
            idx (int): Index of the vocalization

        Returns:
            Audio (torch.Tensor): Audio sample for the index
            Locations (torch.Tensor): A (n_animals, n_samples, n_animals, n_nodes, n_dims) tensor of locations. Index `i`
                along the first dimension indicates that animal index `i` (in dimension 2) is the true location from
                that index and all other locations are random. Unit: arb.
        """

        sound = self.__audio_for_index(idx)
        sound = self.__make_crop(sound, self.crop_length)

        true_label = self.__label_for_index(
            idx
        )  # Shape: (n_animals, n_nodes, n_dims). Unit: dataset unit
        true_label = self.scale_labels(true_label)  # Scale to arb. unit
        num_animals, n_nodes, n_dims = true_label.shape
        labels_with_rands = torch.empty(
            (
                num_animals,
                self.num_negative_samples,
                num_animals,
                true_label.shape[1],
                true_label.shape[2],
            ),
            dtype=true_label.dtype,
        )

        for animal_idx in range(num_animals):
            true_loc = true_label[animal_idx, ...]  # Shape: (n_nodes, n_dims)
            # Sample random locations for each animal. *important* Unit: Arb. unit
            random_labels = self.sample_rand_locations(
                self.num_negative_samples * (num_animals - 1)
            ).reshape(self.num_negative_samples, num_animals - 1, n_nodes, n_dims)

            true_loc = true_loc.unsqueeze(0).expand(
                self.num_negative_samples, -1, -1
            )  # (samps, 1, nodes, dims)
            # Insert at index `animal_idx` in dimension 2 of labels_with_rands
            labels_with_rands[animal_idx, :, :animal_idx, ...] = random_labels[
                :, :animal_idx, ...
            ]
            labels_with_rands[animal_idx, :, animal_idx, ...] = true_loc
            labels_with_rands[animal_idx, :, animal_idx + 1 :, ...] = random_labels[
                :, animal_idx:, ...
            ]

        sound = self.scale_audio(sound)
        # Labels_with_rands is already in arb. unit. Doesn't need scaling

        return sound, labels_with_rands

    def collate(self, batch) -> dict[str, torch.Tensor]:
        """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
        a batch of audio and labels.
        """
        audio, labels = [x[0] for x in batch], [x[1] for x in batch]
        audio = torch.stack(audio)
        labels = torch.stack(labels)

        # Audio should end up with shape (batch, channels, time)
        # Labels should end up with shape (batch, 1 + num_false, n_animals, n_nodes, n_dims)
        return {"audio": audio, "labels": labels}


def get_logical_cores():
    """Gets the number of logical cores available to the program."""
    try:
        # I think this function only exists on linux
        return max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        return max(1, os.cpu_count() - 1)


def build_dataloaders(
    dataset_path: Path,
    index_dir_path: Optional[Path],
    *,
    arena_dims: list[float],
    batch_size: int,
    crop_length: int,
    normalize_data: bool,
    node_names: list[str],
    num_negative_samples: int,
    # num_difficulty_steps: int,
    # min_difficulty: float,
    # max_difficulty: float,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Creates dataloaders for training, validation, and testing.

    Args:
        dataset_path (Path): Path to the dataset. Should be an HDF5 file.
        index_dir_path (Optional[Path]): Path to a directory containing numpy arrays of indices for the train, val, and test sets.
        arena_dims (list[float]): Dimensions of the arena in mm. Used to scale labels.
        batch_size (int): Batch size of all dataloaders
        crop_length (int): Length of audio samples to return.
        normalize_data (bool): Whether to normalize the audio data to have zero mean and unit variance.
        node_names (list[str]): List of skeleton nodes to retrieve from the dataset.
        num_negative_samples (int): Number of negative samples retrieved for each positive example.

    Raises:
        ValueError: If the dataset is improperly formatted

    Returns:
        tuple[DataLoader, DataLoader, Optional[DataLoader]]: Train, validation, and test dataloaders
    """
    train_path = dataset_path
    val_path = dataset_path
    test_path = dataset_path

    data_split_indices = {"train": None, "val": None, "test": None}
    if index_dir_path is not None:
        train_idx_path = index_dir_path / "train_set.npy"
        val_idx_path = index_dir_path / "val_set.npy"
        test_idx_path = index_dir_path / "test_set.npy"
        if not all([train_idx_path.exists(), val_idx_path.exists()]):
            raise ValueError("Index arrays must exist for both train and val sets")
        data_split_indices["train"] = np.load(train_idx_path)
        data_split_indices["val"] = np.load(val_idx_path)
        if test_idx_path.exists():
            data_split_indices["test"] = np.load(test_idx_path)
    else:
        # manually create train/val split
        with h5py.File(train_path, "r") as f:
            if "length_idx" in f:
                dset_size = len(f["length_idx"]) - 1
            else:
                raise ValueError("Improperly formatted dataset")
        full_index = np.arange(dset_size)
        rng = np.random.default_rng(0)
        rng.shuffle(full_index)
        data_split_indices["train"] = full_index[: int(0.8 * dset_size)]
        data_split_indices["val"] = full_index[
            int(0.8 * dset_size) : int(0.9 * dset_size)
        ]
        data_split_indices["test"] = full_index[int(0.9 * dset_size) :]

    training_dataset = VocalizationDataset(
        train_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        num_negative_samples=num_negative_samples,
        index=data_split_indices["train"],
        normalize_data=normalize_data,
        nodes=node_names,
        # difficulty_range=(min_difficulty, max_difficulty),
        # num_difficulty_steps=num_difficulty_steps,
    )

    validation_dataset = VocalizationDataset(
        val_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        num_negative_samples=num_negative_samples,
        crop_randomly=True,
        index=data_split_indices["val"],
        normalize_data=normalize_data,
        nodes=node_names,
        # difficulty_range=(min_difficulty, max_difficulty),
        # num_difficulty_steps=num_difficulty_steps,
    )

    avail_cpus = get_logical_cores()
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=avail_cpus,
        collate_fn=training_dataset.collate,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=avail_cpus,
        shuffle=False,
        collate_fn=validation_dataset.collate,
    )

    test_dataloader = None
    if test_path.exists():
        testdata = VocalizationDataset(
            test_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            crop_randomly=True,
            index=data_split_indices["test"],
            normalize_data=normalize_data,
        )
        test_dataloader = DataLoader(
            testdata,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=testdata.collate,
        )

    return train_dataloader, val_dataloader, test_dataloader


def build_inference_dataset(
    dataset_path: Path,
    index_path: Optional[Path],
    *,
    arena_dims: list[float],
    crop_length: int,
    normalize_data: bool,
    node_names: list[str],
    batch_size: int,
) -> DataLoader:
    """Constructs a single dataset for performing inference on a dataset.

    Args:
        dataset_path (Path): Path to the dataset. Should be an HDF5 file.
        index_path (Optional[Path]): Path to a numpy array of indices for the dataset.
        arena_dims (list[float]): Dimensions of the arena in mm. Used to scale labels.
        batch_size (int): Batch size of the dataloader
        crop_length (int): Length of audio samples to return.
        normalize_data (bool): Whether to normalize the audio data to have zero mean and unit variance.
        node_names (list[str]): List of skeleton nodes to retrieve from the dataset.

    Returns:
        VocalizationDataset: The dataset for inference
    """

    if index_path is not None:
        indices = np.load(index_path)
    else:
        indices = None

    inference_dataset = VocalizationDataset(
        dataset_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        crop_randomly=True,  # Todo: Experiment with this
        inference=False,
        index=indices,
        normalize_data=normalize_data,
        nodes=node_names,
        num_negative_samples=0,
    )

    loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=get_logical_cores(),
        shuffle=False,
        collate_fn=inference_dataset.collate,
    )
    return loader
