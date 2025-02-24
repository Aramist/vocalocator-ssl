"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.io import wavfile
from scipy.spatial import KDTree
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio import functional as AF


class VocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        crop_length: int,
        false_locations_per_sample: int = 1,
        *,
        nodes: Optional[list[str]] = None,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
        index: Optional[np.ndarray] = None,
        normalize_data: bool = True,
        difficulty_range: Optional[tuple[float, float]] = None,
        num_difficulty_steps: int = 80_000,
    ):
        """
        Args:
            datapath (Path): Path to HDF5 dataset
            inference (bool, optional): When true, data will be cropped deterministically. Defaults to False.
            crop_length (int): Length of audio samples to return.
            arena_dims (Optional[Union[np.ndarray, Tuple[float, float]]], optional): Dimensions of the arena in mm. Used to scale labels.
            index (Optional[np.ndarray], optional): An array of indices to use for this dataset. Defaults to None, which will use the full dataset
        """
        if isinstance(datapath, str):
            datapath = Path(datapath)
        self.datapath = datapath
        dataset = h5py.File(self.datapath, "r")
        # dataset cannot exist as a member of the object until after pytorch has cloned and
        # spread the dataset objects across multiple processes.
        # This is because h5py handles cannot be pickled and pytorch uses pickle under the hood
        # I get around this by re-initializing the h5py.File lazily in __getitem__
        self.dataset: Optional[h5py.File] = None

        if not isinstance(arena_dims, np.ndarray):
            arena_dims = np.array(arena_dims).astype(np.float32)

        if "length_idx" not in dataset and "rir_length_idx" not in dataset:
            raise ValueError("Improperly formatted dataset")

        if "audio" not in dataset:
            raise ValueError("Improperly formatted dataset")

        self.inference: bool = inference
        self.arena_dims: np.ndarray = arena_dims
        self.crop_length: int = crop_length
        self.num_false_locations: int = false_locations_per_sample
        self.index: Optional[np.ndarray] = index
        self.normalize_data: bool = normalize_data
        self.difficulty_range: Optional[tuple[float, float]] = difficulty_range
        self.source_point_index: Optional[KDTree] = None
        self.num_difficulty_steps: int = num_difficulty_steps
        self.cur_difficulty_step: int = 0
        if self.index is None:
            self.index = np.arange(len(dataset["length_idx"]) - 1)
        self.length: int = len(self.index)

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

        self.sample_vocalizations = []
        if self.difficulty_range is not None:
            source_points = dataset["locations"][..., self.node_indices[0], :]
            if len(source_points.shape) == 3:
                # flatten multi-animal dimension
                source_points = source_points.reshape(-1, source_points.shape[-1])
            self.source_point_index = KDTree(source_points)

        dataset.close()  # The dataset cannot be pickled by pytorch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            # Lazily recreate the dataset object in the child process
            self.dataset = h5py.File(self.datapath, "r")
        true_idx = self.index[idx]
        return self.__processed_data_for_index__(true_idx)

    def __make_crop(self, audio: torch.Tensor, crop_length: int):
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

    def __label_for_index(self, idx: int) -> Optional[np.ndarray]
        """Gets the ground truth source location for the vocalization at the given index

        Args:
            idx (int): Index of the vocalization in the dataset

        Returns:
            np.ndarray: The source location of the vocalization, if available. Shape: (n_node, n_dim)
        """

        if "locations" not in self.dataset:
            return None

        locs = self.dataset["locations"][idx, ..., self.node_indices, :]
        locs = torch.from_numpy(locs.astype(np.float32))
        return locs

    @property
    def difficulty(self):
        """Returns the current difficulty level, which is a value between 0 and 1."""
        if self.difficulty_range is None:
            return None

        return self.cur_difficulty_step / self.num_difficulty_steps

    def step_difficulty(self):
        self.cur_difficulty_step = min(
            self.cur_difficulty_step + 1, self.num_difficulty_steps
        )

    def get_current_sample_difficulty(self):
        """Returns the range about which negative samples are sampled based on the
        current difficulty level.
        """
        if self.difficulty_range is None:
            return None

        dist_min, dist_max = min(self.difficulty_range), max(self.difficulty_range)

        # At difficulty=1, the max distance should be dist_min
        # At difficulty=0, the max distance should be dist_max
        sample_dist = dist_max + (dist_min - dist_max) * self.difficulty
        return sample_dist

    def sample_false_locations(self, positive_location: np.ndarray):
        """Samples false locations for a batch of audio samples. False locations are
        sampled uniformly from the dataset to match the distribution of true locations.

        Args: positive_location (np.ndarray): The true location for which false locations
            are being sampled. Shape: (D,)
        """
        num_dims = int(len(positive_location) // len(self.node_indices))
        num_samples = self.num_false_locations
        sample_dist = self.get_current_sample_difficulty()
        if sample_dist is None:
            indices = np.arange(self.length)
        else:
            indices = self.source_point_index.query_ball_point(
                positive_location[:num_dims], sample_dist
            )[0]

        if len(indices) > num_samples:
            samples = self.rng.choice(indices, num_samples, replace=False)
        else:
            _, samples = self.source_point_index.query(
                positive_location[:num_dims], k=num_samples
            )
            samples = samples.squeeze()

        return [self.__label_for_index(idx) for idx in samples]

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

        if labels is not None:
            # Shift range to [-1, 1], but keep units same across dimensions
            # Origin remains at center of arena
            scale_factor = self.arena_dims.max() / 2
            labels = labels / scale_factor

        return labels

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(idx)
        sound = self.__make_crop(sound, self.crop_length)
        locations = [self.__label_for_index(idx)]

        if len(locations[0].shape) == 2:  # (nodes, dims)
            locations.extend(self.sample_false_locations(locations[0].numpy()))
            locations = torch.stack(locations)  # (num_false + 1, nodes, dims)
        else:
            # we have a dyadic dataset
            locations = locations[0]  # (mice, nodes, dims)

        sound, locations = self.scale_audio(sound), self.scale_labels(locations)

        return sound, locations

    def collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
        a batch of audio and labels.
        """
        audio, labels = [x[0] for x in batch], [x[1] for x in batch]
        audio = torch.stack(audio)
        if labels[0] is not None:
            labels = torch.stack(labels)
            labels = labels.view(labels.shape[0], labels.shape[1], -1)
        else:
            labels = [None] * len(audio)

        # Audio should end up with shape (batch, channels, time)
        # Labels should end up with shape (batch, 1 + num_false, num_nodes * num_dimensions)
        return {"audio": audio, "labels": labels}


def build_dataloaders(
    path_to_data: Union[Path, str], config: dict, index_dir: Optional[Path]
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # Construct Dataset objects.
    if not isinstance(path_to_data, Path):
        path_to_data = Path(path_to_data)

    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"]["CROP_LENGTH"]
    normalize_data = config["DATA"].get("NORMALIZE_DATA", True)
    node_names = config["DATA"].get("NODES_TO_LOAD", None)

    min_difficulty = config["DATA"].get("MIN_DIFFICULTY", 300)  # 30cm radius
    max_difficulty = config["DATA"].get("MAX_DIFFICULTY", 50)  # 5cm radius
    num_difficulty_steps = config["DATA"].get("NUM_DIFFICULTY_STEPS", 80_000)

    num_fakes = config["DATA"].get("NUM_FAKE_LOCATIONS", 1)

    vocalization_dir = config["DATA"].get(
        "VOCALIZATION_DIR",
        None,
    )

    index_arrays = {"train": None, "val": None, "test": None}
    if index_dir is not None:
        index_arrays["train"] = np.load(index_dir / "train_set.npy")
        index_arrays["val"] = np.load(index_dir / "val_set.npy")
        if (index_dir / "test_set.npy").exists():
            index_arrays["test"] = np.load(index_dir / "test_set.npy")

    try:
        # I think this function only exists on linux
        avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        avail_cpus = max(1, os.cpu_count() - 1)

    if path_to_data.is_dir():
        train_path = path_to_data / "train_set.h5"
        val_path = path_to_data / "val_set.h5"
        test_path = path_to_data / "test_set.h5"
    else:
        train_path = path_to_data
        val_path = path_to_data
        test_path = path_to_data
        if index_dir is None:
            # manually create train/val split
            with h5py.File(train_path, "r") as f:
                if "length_idx" in f:
                    dset_size = len(f["length_idx"]) - 1
                elif "rir_length_idx" in f:
                    dset_size = len(f["rir_length_idx"]) - 1
                else:
                    raise ValueError("Improperly formatted dataset")
            full_index = np.arange(dset_size)
            rng = np.random.default_rng(0)
            rng.shuffle(full_index)
            index_arrays["train"] = full_index[: int(0.8 * dset_size)]
            index_arrays["val"] = full_index[
                int(0.8 * dset_size) : int(0.9 * dset_size)
            ]
            index_arrays["test"] = full_index[int(0.9 * dset_size) :]

    traindata = VocalizationDataset(
        train_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        false_locations_per_sample=num_fakes,
        index=index_arrays["train"],
        normalize_data=normalize_data,
        sample_vocalization_dir=vocalization_dir,
        nodes=node_names,
        difficulty_range=(min_difficulty, max_difficulty),
        num_difficulty_steps=num_difficulty_steps,
    )

    valdata = VocalizationDataset(
        val_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        false_locations_per_sample=num_fakes,
        inference=True,
        index=index_arrays["val"],
        normalize_data=normalize_data,
        sample_vocalization_dir=vocalization_dir,
        nodes=node_names,
        difficulty_range=(min_difficulty, max_difficulty),
        num_difficulty_steps=num_difficulty_steps,
    )

    train_dataloader = DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=avail_cpus,
        collate_fn=traindata.collate,
    )

    val_dataloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=avail_cpus,
        shuffle=False,
        collate_fn=valdata.collate,
    )

    test_dataloader = None
    if test_path.exists():
        testdata = VocalizationDataset(
            test_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            inference=True,
            index=index_arrays["test"],
            normalize_data=normalize_data,
            sample_vocalization_dir=vocalization_dir,
        )
        test_dataloader = DataLoader(
            testdata,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=testdata.collate,
        )

    return train_dataloader, val_dataloader, test_dataloader
