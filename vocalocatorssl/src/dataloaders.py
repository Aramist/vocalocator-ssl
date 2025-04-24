"""
Framework for constructing Datasets and DataLoaders for training and inference
"""

import os
import typing as tp
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.spatial import KDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def sample_pairwise_distances(pts: np.ndarray, num_samples=1000) -> np.ndarray:
    """Samples random points from the dataset and estimates the distribution of pairwise distances
    based on the sample.

    Args:
        pts (np.ndarray): Array of points. Shape: (n_samples, n_dims)

    Returns:
        np.ndarray: Flat array of pairwise distances. Shape: (n_samples * (n_samples - 1) / 2,)
    """

    # Sample random points from the dataset
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(pts), num_samples, replace=False)
    pts = pts[sample_indices, :]

    magsqr = np.sum(pts**2, axis=1)  # (n_samples,)
    dot_p = np.einsum("ad,bd->ab", pts, pts)  # (n_samples, n_samples)

    dist_matrix = magsqr[:, None] + magsqr[None, :] - 2 * dot_p
    dist_matrix = np.sqrt(dist_matrix)

    # Take lower triangle without diag
    dist_matrix = dist_matrix[np.tril_indices(len(pts), -1)]
    dist_matrix = dist_matrix.reshape(-1)
    return dist_matrix


def collate(batch) -> dict[str, torch.Tensor]:
    """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
    a batch of audio and labels.
    """
    audio, labels = [x[0] for x in batch], [x[1] for x in batch]
    audio = torch.stack(audio)
    labels = torch.stack(labels)

    # Audio should end up with shape (batch, channels, time)
    # Labels should end up with shape (batch, 1 + num_false, n_animals, n_nodes, n_dims)
    return {"audio": audio, "labels": labels}


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
        construct_search_tree: bool = True,
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
            construct_search_tree (bool, optional): Whether to construct a KDTree for the dataset. This is used for sampling negative locations. Defaults to True.
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
        self.length: int = len(self.index)
        self.inverse_index = {v: k for k, v in enumerate(self.index)}
        self.rng: np.random.Generator

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

        multi_animal = len(dataset["locations"].shape) > 3
        if multi_animal and construct_search_tree:  # only needed for training
            # Expected shape: (dataset_size, n_animals, n_nodes, n_dims)
            animal_configurations: np.ndarray = dataset["locations"][
                ..., self.node_indices[0], :
            ]

            animal_configurations = animal_configurations[self.index, ...]
            animal_configurations = animal_configurations[
                ..., :2
            ]  ## experimental: only use x and y for one node
            centroids = animal_configurations.mean(axis=1).reshape(
                len(self.index), -1
            )  # (dataset_size, n_node * n_dim)
            self.search_tree = KDTree(centroids)
            # Estimate the distribution of distances between points
            pairwise_distances = sample_pairwise_distances(centroids)
            pairwise_distances.sort()
            # 200 is the granularity of the inverse CDF
            indices = (np.linspace(0.1, 0.9, 200) * len(pairwise_distances)).astype(int)
            indices = np.clip(indices, 0, len(pairwise_distances) - 1)
            self.pairwise_distance_quantiles = pairwise_distances[indices]

        dataset.close()  # The dataset cannot be pickled by pytorch

    def __init_inside_subproc(self):
        """This function is called in the child process to reinitialize the dataset object.
        This is necessary because h5py handles cannot be pickled and pytorch uses pickle under the hood.
        """
        self.dataset = h5py.File(self.datapath, "r")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.rng = np.random.default_rng(0)
        else:
            self.rng = np.random.default_rng(worker_info.id)

    def __len__(self):
        return self.length

    def __getitem__(
        self, idx_and_difficulty: tuple[int, Optional[float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wraps the __processed_data_for_index__ method to provide lazy loading
        of the h5py.File object and abstraction of the dataset index.

        Args:
            idx_and_difficulty (tuple[int, Optional[float]]): Index of the vocalization and difficulty level
        The difficulty level is used to sample negative locations from the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The audio and locations (+ and -) for
        the index. Audio is of shape (n_samples, n_channels).
        Locations are of shape (n_negative + 1, n_animals, n_nodes, n_dims)
        """
        if not isinstance(idx_and_difficulty, tuple):
            raise ValueError(
                "Idx_and_difficulty must be a tuple of (index, float). If using a DataLoader, pass a custom DifficultySampler instance to generate indices"
            )

        if self.dataset is None or not hasattr(self, "rng"):
            # Lazily recreate the dataset object in the child process
            self.__init_inside_subproc()
        orig_idx, difficulty = idx_and_difficulty
        true_idx = self.index[orig_idx]
        return self.__processed_data_for_index__(true_idx, difficulty)

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

    def sample_negative_location(
        self, idx: int, difficulty: Optional[float], *, n: int = 1
    ) -> torch.Tensor:
        """Samples a negative frame from the dataset for index `idx` based on the
        current difficulty level. This is done by computing a distribution over
        the dataset based on each frame's difficulty and sampling from it to obtain
        an index.

        Args:
            idx (int): Current index in [0, len(self.dataset)). This function guarantees the sampled index will be distinct
            difficulty (Optional[float]): Difficulty level of the current index. Smaller values are more difficult. Should be between 0 and 1 (0 is most difficult).
            n (int): Number of negative samples to return. Defaults to 1.

        Returns:
            torch.Tensor: Animal poses for the negative frame. Shape: (n_requested, n_animals, n_nodes, n_dims)
        """
        if n == 0:
            n_animals, n_nodes, n_dims = self.__label_for_index(idx).shape
            return torch.empty((0, n_animals, n_nodes, n_dims))
        if difficulty is None:
            choices = np.delete(self.index, self.inverse_index[idx])
            neg_idx = self.rng.choice(choices, size=n, replace=False)
            return torch.stack([self.__label_for_index(i) for i in neg_idx], dim=0)

        search_radius = self.pairwise_distance_quantiles[
            int(difficulty * (len(self.pairwise_distance_quantiles) - 1))
        ]
        ## Experimental: use only x and y coordinates to make this faster
        search_center = self.__label_for_index(idx).mean(dim=0).reshape(-1)[:2]

        # Sample a point within the search radius
        points_in_radius = self.search_tree.query_ball_point(
            search_center.numpy(), search_radius
        )
        if len(points_in_radius) < self.num_negative_samples:
            # get more entropy by sampling nearest neighbors
            _, points_in_radius = self.search_tree.query(
                search_center.numpy(), k=num_negative_samples * 2
            )
            points_in_radius = points_in_radius[1:]  # Exclude the point itself

        # Sample a random point from the points in the radius
        sample_idx = self.rng.choice(points_in_radius, size=n, replace=False)
        # The indices in the KDTree are already in the range [0, len(self.index))
        return torch.stack([self.__label_for_index(s_i) for s_i in sample_idx], dim=0)

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

    def scale_audio(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Scales the inputs to have zero mean and unit variance (across all channels)."""

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
        self, idx: int, difficulty: Optional[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the audio and locations from a given index in the dataset. This function
        should not be called directly.

        Args:
            idx (int): Index of the vocalization
            difficulty (Optional[float]): Difficulty level of the current index. Should be between 0 and 1 (0 is most difficult).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The audio and locations (+ and -) for
        the index. Audio is of shape (n_samples, n_channels).
        Locations are of shape (n_negative + 1, n_animals, n_nodes, n_dims)
        """

        sound = self.__audio_for_index(idx)
        sound = self.__make_crop(sound, self.crop_length)
        positive_location = self.__label_for_index(idx)
        negative_locations = self.sample_negative_location(
            idx, difficulty, n=self.num_negative_samples
        )
        locations = torch.cat(
            [positive_location.unsqueeze(0), negative_locations], dim=0
        )
        sound, locations = self.scale_audio(sound), self.scale_labels(locations)

        return sound, locations


class DifficultySampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: VocalizationDataset,
        *,
        difficulty_range: Optional[tuple[float, float]] = None,
        num_difficulty_steps: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
    ):
        super().__init__()
        self.length: int = len(dataset)
        self.shuffle: bool = shuffle
        self.difficulty_range: Optional[tuple[float, float]] = difficulty_range
        self.num_difficulty_steps: Optional[int] = num_difficulty_steps
        self.cur_difficulty_step: int = 0
        self.difficulty_step_increment = 1
        self.rng = np.random.default_rng(seed)

    @property
    def difficulty(self) -> Optional[float]:
        """Returns the current difficulty level, which is a value between 0 and 1."""
        if self.difficulty_range is None:
            return None

        return self.cur_difficulty_step / self.num_difficulty_steps

    def step_difficulty(self):
        if self.num_difficulty_steps is None:
            return

        self.cur_difficulty_step = min(
            self.cur_difficulty_step + self.difficulty_step_increment,
            self.num_difficulty_steps,
        )

    def sample_difficulty(self) -> float:
        """From the current difficulty level, sample a difficulty index from the beta distribution.

        Returns:
            float: Sampled difficulty index between 0 and 1
        """
        if self.difficulty_range is None:
            return None

        min_d, max_d = self.difficulty_range
        cur_difficulty = (
            self.difficulty * (max_d - min_d) + min_d
        )  # float between min and max difficulty
        beta = 10**cur_difficulty
        alpha = (
            1 / beta
        )  # Smaller values of cur_difficulty yield samples of the beta distribution
        # closer to zero, which corresponds to early (low spread) indices in the sorting
        # index
        sample = self.rng.beta(alpha, beta, 1).item()  # Sampled value between 0 and 1
        return sample

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> tp.Iterator[tuple[int, Optional[float]]]:
        shuffled_idx = np.arange(self.length)
        if self.shuffle:
            self.rng.shuffle(shuffled_idx)

        for i in shuffled_idx:
            yield i, self.sample_difficulty()
            self.step_difficulty()


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
    num_difficulty_steps: int,
    min_difficulty: float,
    max_difficulty: float,
    sampler_seed: int = 0,
    num_val_negative_samples: int | None = None,
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
    )

    validation_dataset = VocalizationDataset(
        val_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        num_negative_samples=num_negative_samples
        if num_val_negative_samples is None
        else num_val_negative_samples,
        crop_randomly=True,
        index=data_split_indices["val"],
        normalize_data=normalize_data,
        nodes=node_names,
        construct_search_tree=False,
    )

    difficulty_range = (
        None
        if max_difficulty is None or min_difficulty is None
        else (min_difficulty, max_difficulty)
    )
    train_difficulty_steps = (
        num_difficulty_steps * batch_size if difficulty_range is not None else None
    )
    avail_cpus = get_logical_cores()
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=avail_cpus,
        collate_fn=collate,
        sampler=DifficultySampler(
            training_dataset,
            difficulty_range=difficulty_range,
            num_difficulty_steps=train_difficulty_steps,
            shuffle=True,
            seed=sampler_seed,
        ),
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=avail_cpus,
        sampler=DifficultySampler(
            validation_dataset,
            shuffle=False,
            seed=sampler_seed,
        ),
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
            construct_search_tree=False,
        )
        test_dataloader = DataLoader(
            testdata,
            batch_size=batch_size,
            collate_fn=collate,
            sampler=DifficultySampler(
                testdata,
                difficulty_range=None,
                num_difficulty_steps=None,
                shuffle=False,
            ),
        )

    return train_dataloader, val_dataloader, test_dataloader


def build_inference_dataset(
    dataset_path: Path,
    index_path: Optional[Path],
    *,
    arena_dims: list[float],
    batch_size: int,
    crop_length: int,
    normalize_data: bool,
    node_names: list[str],
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
        index=indices,
        normalize_data=normalize_data,
        nodes=node_names,
        num_negative_samples=0,
    )

    loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        num_workers=get_logical_cores(),
        collate_fn=collate,
        sampler=DifficultySampler(
            inference_dataset,
            difficulty_range=None,
            num_difficulty_steps=None,
            shuffle=False,
        ),
    )
    return loader
