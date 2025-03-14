"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch

# from scipy.spatial import KDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class VocalizationDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        *,
        crop_length: int,
        num_negative_samples: int = 1,
        nodes: Optional[list[str]] = None,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, list[float]]] = None,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
        # difficulty_range: Optional[tuple[float, float]] = None,
        # num_difficulty_steps: int = 80_000,
    ):
        """
        Args:
            datapath (Path): Path to HDF5 dataset
            inference (bool, optional): When true, data will be cropped deterministically. Defaults to False.
            crop_length (int): Length of audio samples to return.
            arena_dims (Optional[Union[np.ndarray, Tuple[float, float]]], optional): Dimensions of the arena in mm. Used to scale labels.
            index (Optional[np.ndarray], optional): An array of indices to use for this dataset. Defaults to None, which will use the full dataset
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

        self.inference: bool = inference
        self.arena_dims: np.ndarray = arena_dims
        self.crop_length: int = crop_length
        self.num_negative_samples: int = num_negative_samples
        self.index: np.ndarray = (
            index if index is not None else np.arange(len(dataset["length_idx"]) - 1)
        )
        self.normalize_data: bool = normalize_data
        # self.difficulty_range: Optional[tuple[float, float]] = difficulty_range
        # self.source_point_index: Optional[KDTree] = None
        # self.num_difficulty_steps: int = num_difficulty_steps
        # self.cur_difficulty_step: int = 0
        self.length: int = len(self.index)
        self.inverse_index = {v: k for k, v in enumerate(self.index)}

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

        # if self.difficulty_range is not None:
        #     source_points = dataset["locations"][..., self.node_indices[0], :]
        #     if len(source_points.shape) == 3:
        #         # flatten multi-animal dimension
        #         source_points = source_points.reshape(-1, source_points.shape[-1])
        #     self.source_point_index = KDTree(source_points)

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
        Locations are of shape (n_negative + 1, n_mice, n_nodes, n_dims)
        """
        if self.dataset is None:
            # Lazily recreate the dataset object in the child process
            self.dataset = h5py.File(self.datapath, "r")
        true_idx = self.index[idx]
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
        if self.inference:
            range_start = 0  # Ensure the output is deterministic at inference time
        else:
            range_start = self.rng.integers(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

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
            torch.Tensor: The source location of the vocalization, if available. Shape: (n_mice, n_node, n_dim)
        """

        locs = self.dataset["locations"][idx, ..., self.node_indices, :]
        locs = torch.from_numpy(locs.astype(np.float32))
        return locs

    def sample_negative_location(self, idx: int) -> torch.Tensor:
        """Samples a negative ground truth from the dataset. This is defined as a set of animal
        positions from a different time point than the positive sample.

        Args:
            idx (int): True index of the positive sample in the dataset

        Returns:
            torch.Tensor: The negative sample. Shape: (n_mice, n_node, n_dim)
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
            torch.Tensor: Batch of scaled, random locations. Shape: (num, n_node, n_dim)
        """
        if self.dataset is None:
            # Lazily recreate the dataset object in the child process
            self.dataset = h5py.File(self.datapath, "r")

        indices = self.rng.choice(len(self.dataset), num, replace=True)
        labels = torch.stack([self.__label_for_index(idx) for idx in indices])
        # Labels shape: (num, n_mice, n_nodes, n_dims)
        # May have more than one mouse in each location, randomly choose one per location
        mouse_choices = torch.from_numpy(
            self.rng.integers(0, labels.shape[1], size=(num,))
        )
        labels = labels[torch.arange(num), mouse_choices, ...]
        return labels  # Shape: (num, n_nodes, n_dims)

    # TODO: Decide how to scale difficulty in multi-animal datasets
    # @property
    # def difficulty(self):
    #     """Returns the current difficulty level, which is a value between 0 and 1."""
    #     if self.difficulty_range is None:
    #         return None

    #     return self.cur_difficulty_step / self.num_difficulty_steps

    # def step_difficulty(self):
    #     self.cur_difficulty_step = min(
    #         self.cur_difficulty_step + 1, self.num_difficulty_steps
    #     )

    # def get_current_sample_difficulty(self):
    #     """Returns the range about which negative samples are sampled based on the
    #     current difficulty level.
    #     """
    #     if self.difficulty_range is None:
    #         return None

    #     dist_min, dist_max = min(self.difficulty_range), max(self.difficulty_range)

    #     # At difficulty=1, the max distance should be dist_min
    #     # At difficulty=0, the max distance should be dist_max
    #     sample_dist = dist_max + (dist_min - dist_max) * self.difficulty
    #     return sample_dist

    # def sample_false_locations(self, positive_location: np.ndarray):
    #     """Samples false locations for a batch of audio samples. False locations are
    #     sampled uniformly from the dataset to match the distribution of true locations.

    #     Args: positive_location (np.ndarray): The true location for which false locations
    #         are being sampled. Shape: (D,)
    #     """
    #     num_dims = int(len(positive_location) // len(self.node_indices))
    #     num_samples = self.num_false_locations
    #     sample_dist = self.get_current_sample_difficulty()
    #     if sample_dist is None:
    #         indices = np.arange(self.length)
    #     else:
    #         indices = self.source_point_index.query_ball_point(
    #             positive_location[:num_dims], sample_dist
    #         )[0]

    #     if len(indices) > num_samples:
    #         samples = self.rng.choice(indices, num_samples, replace=False)
    #     else:
    #         _, samples = self.source_point_index.query(
    #             positive_location[:num_dims], k=num_samples
    #         )
    #         samples = samples.squeeze()

    #     return [self.__label_for_index(idx) for idx in samples]
    # END TODO

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
        Locations are of shape (n_negative + 1, n_mice, n_nodes, n_dims)
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

    def collate(self, batch) -> dict[str, torch.Tensor]:
        """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
        a batch of audio and labels.
        """
        audio, labels = [x[0] for x in batch], [x[1] for x in batch]
        audio = torch.stack(audio)
        labels = torch.stack(labels)

        # Audio should end up with shape (batch, channels, time)
        # Labels should end up with shape (batch, 1 + num_false, n_mice, n_nodes, n_dims)
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
        inference=True,
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
            inference=True,
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
) -> VocalizationDataset:
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
        inference=True,
        index=indices,
        normalize_data=normalize_data,
        nodes=node_names,
        num_negative_samples=0,
    )
    return inference_dataset
