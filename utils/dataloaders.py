import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from gerbilizer.training.dataloaders import GerbilVocalizationDataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info


class GerbilizerDatasetWrapper(Dataset):
    def __init__(
        self,
        datapaths: List[Path],
        crop_length: int,
        arena_dims: Tuple[float, float],
        make_cps: bool,
    ):
        print(f"initializing GerbilizerDatasetWrapper...  {len(datapaths)} files")
        self.datapaths = datapaths
        self.crop_length = crop_length
        self.arena_dims = arena_dims
        self.make_cps = make_cps
        self.active_dataset: Optional[GerbilVocalizationDataset] = None
        self.active_dataset_ordering = np.arange(len(self.datapaths))

        worker_info = get_worker_info()
        if worker_info is None:
            random_seed = 0
        else:
            random_seed = worker_info.id

        np.random.default_rng(random_seed).shuffle(self.active_dataset_ordering)
        # Current active file index within the ordering
        self.active_dataset_idx: int = 0
        # Number of samples returned in the current file
        self.returned_samples: int = 0
        self.compute_length()
        self.load_dataset()

    def _min_max_scale(self, audio: Tensor) -> Tensor:
        """Min max scales the input audio to [-1, 1] to match the range of tanh"""
        is_batched = audio.ndim == 3  # batch, samples, channels
        if is_batched:
            min_vals = (
                audio.min(dim=1, keepdim=True).values.min(dim=2, keepdim=True).values
            )
            max_vals = (
                audio.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
            )
        else:
            min_vals = audio.min()
            max_vals = audio.max()
        audio = (audio - min_vals) / (max_vals - min_vals)  # [0, 1]
        audio = audio * 2 - 1  # [-1, 1]
        return audio

    def load_dataset(self):
        if self.active_dataset is not None:
            del self.active_dataset  # Should close the h5 on deallocation

        # in theory, active_dataset_idx should never go out of bounds, but just in case
        self.active_dataset_idx = self.active_dataset_idx % len(self.datapaths)
        self.active_dataset = GerbilVocalizationDataset(
            datapath=str(
                self.datapaths[self.active_dataset_ordering[self.active_dataset_idx]]
            ),
            inference=True,
            crop_length=self.crop_length,
            arena_dims=self.arena_dims,
            make_cps=self.make_cps,
        )
        self.active_dataset_idx += 1
        self.returned_samples = 0

    def compute_length(self):
        self.length = 0
        for datapath in self.datapaths:
            with h5py.File(datapath, "r") as ctx:
                self.length += len(ctx["len_idx"]) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Ensure the index is within the bounds of the dataset
        dataset_length = len(self.active_dataset)
        idx = idx % dataset_length

        # Load the data
        data, cps = self.active_dataset[idx]
        data = self._min_max_scale(data)  # important because the generator uses tanh

        data = data, cps

        # See if a new dataset needs to be loaded
        self.returned_samples += 1
        if self.returned_samples >= dataset_length:
            self.load_dataset()

        return data


def build_dataloader(data_directory: Path, config: dict) -> DataLoader:
    """Build a dataloader from a directory of HDF5 files
    The HDF5 files are expected to contain unlabeled vocalizations
    """
    h5_files = list(data_directory.glob("*.h5"))

    crop_length = config["DATA"]["CROP_LENGTH"]
    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    make_cps = config["DATA"]["MAKE_CPS"]

    dset = GerbilizerDatasetWrapper(
        datapaths=h5_files,
        crop_length=crop_length,
        arena_dims=arena_dims,
        make_cps=make_cps,
    )
    # Always leave one core free for the main process
    avail_workers = max(1, len(os.sched_getaffinity(0)) - 1)

    dloader = DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=avail_workers,
    )

    return dloader
