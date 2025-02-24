import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocalocator.training.augmentations import build_augmentations
from vocalocator.training.configs import update_recursively

from .dataloaders import VocalizationDataset, build_dataloaders
from .model import Wavenet

DEFAULT_CONFIG = {
    "OPTIMIZATION": {
        "NUM_WEIGHT_UPDATES": 500_000,
        "WEIGHT_UPDATES_PER_EPOCH": 10_000,
        "NUM_WARMUP_STEPS": 10_000,
        "OPTIMIZER": "SGD",
        "MOMENTUM": 0.7,
        "WEIGHT_DECAY": 1e-05,
        "CLIP_GRADIENTS": True,
        "INITIAL_LEARNING_RATE": 0.003,
    },
    "ARCHITECTURE": "CorrSimpleNetwork",
    "MODEL_PARAMS": {},
    "DATA": {
        "NUM_MICROPHONES": 24,
        "CROP_LENGTH": 8192,
        "ARENA_DIMS": [615, 615, 425],
        "ARENA_DIMS_UNITS": "MM",
        "SAMPLE_RATE": 250000,
        "BATCH_SIZE": 32,
        "NUM_FAKE_LOCATIONS": 4,
        "MIN_DIFFICULTY": 300,
        "MAX_DIFFICULTY": 30,
    },
    "AUGMENTATIONS": {
        "AUGMENT_DATA": True,
        "INVERSION": {"PROB": 0.5},
        "NOISE": {"MIN_SNR": 3, "MAX_SNR": 15, "PROB": 0.5},
        "MASK": {"PROB": 0.5, "MIN_LENGTH": 256, "MAX_LENGTH": 512},
    },
}

logger = None


def get_mem_usage():
    if not torch.cuda.is_available():
        return ""
    used_gb = torch.cuda.max_memory_allocated() / (2**30)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
    torch.cuda.reset_peak_memory_stats()
    return "Max mem. usage: {:.2f}/{:.2f}GiB".format(used_gb, total_gb)


def train(
    config: dict,
    data_path: Path,
    save_directory: Optional[Path] = None,
):
    global logger
    if not save_directory:
        print("No save directory specified, not saving model checkpoints.")
        log_path = "train_log.log"
    else:
        log_path = save_directory / "train_log.log"
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available!")

    # Log to stdout and file
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    cfg = DEFAULT_CONFIG.copy()
    config = update_recursively(config, cfg)
    model: Wavenet = Wavenet(config)
    model.cuda()
    augmentor = build_augmentations(config)

    train_dloader, val_dloader, test_dloader = build_dataloaders(
        data_path, config, None
    )

    if save_directory is not None:
        with open(save_directory / "config.json", "w") as ctx:
            json.dump(config, ctx)
        index_dir = save_directory / "indices"
        index_dir.mkdir(exist_ok=True)
        np.save(save_directory / "train_idx.npy", train_dloader.dataset.index)
        np.save(save_directory / "val_idx.npy", val_dloader.dataset.index)
        if test_dloader is not None:
            np.save(save_directory / "test_idx.npy", test_dloader.dataset.index)

    opt_config = config["OPTIMIZATION"]
    total_num_updates = opt_config["NUM_WEIGHT_UPDATES"]

    opt = optim.SGD(
        model.parameters(),
        lr=opt_config["INITIAL_LEARNING_RATE"],
        momentum=0.7,
    )

    n_warmup_steps = opt_config["NUM_WARMUP_STEPS"]
    warmup_sched = optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, n_warmup_steps)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=(total_num_updates - n_warmup_steps),
        eta_min=0,
        verbose=False,
    )
    sched = optim.lr_scheduler.SequentialLR(
        opt, [warmup_sched, cosine_sched], milestones=[n_warmup_steps]
    )
    log_interval = 500

    proc_times = deque(maxlen=30)

    epoch = 0
    mb_per_epoch = opt_config["WEIGHT_UPDATES_PER_EPOCH"]
    n_mb_processed_cur_epoch = 0
    n_mb_procd = lambda: epoch * mb_per_epoch + n_mb_processed_cur_epoch
    while n_mb_procd() < total_num_updates:

        while n_mb_processed_cur_epoch < mb_per_epoch:
            # Training period
            for batch in train_dloader:
                if n_mb_processed_cur_epoch >= mb_per_epoch:
                    break
                opt.zero_grad()

                audio, labels = batch["audio"], batch["labels"]
                rand_perm = torch.randperm(labels.shape[1])
                audio = audio.cuda()
                labels = labels[:, rand_perm, ...].cuda()
                class_label = (
                    rand_perm.numpy().argmin()
                )  # location of the true label in the shuffled array

                aug_audio = augmentor(audio)
                embedded_audio = model.embed_audio(aug_audio)
                embedded_locations = model.embed_location(labels)

                scores = model.score_pairs(embedded_audio, embedded_locations)
                targets = torch.tensor([class_label] * audio.shape[0]).cuda()
                loss = F.cross_entropy(scores, targets)

                loss.backward()

                proc_times.append(time.time())

                opt.step()
                sched.step()
                train_dloader.dataset.step_difficulty()
                n_mb_processed_cur_epoch += 1
                # Log progress
                if (n_mb_procd() % log_interval) == 0:
                    logger.info(get_mem_usage())
                    processing_speed = len(proc_times) / (
                        proc_times[-1] - proc_times[0]
                    )
                    logger.info(
                        f"Progress: {n_mb_procd()} / {total_num_updates} weight updates."
                    )
                    logger.info(
                        f"Current sampling difficulty: {train_dloader.dataset.difficulty:.2f} / 1.0"
                    )
                    # Losses are in the order time, freq, kl, features, discriminator, adversarial
                    loss = loss.detach().cpu().item()
                    logger.info(f"Loss: {loss:.2f}")
                    logger.info(
                        f"Speed: {processing_speed*audio.shape[0]:.1f} vocalizations per second"
                    )
                    eta = (total_num_updates - n_mb_procd()) / processing_speed
                    eta_hours = int(eta // 3600)
                    eta_minutes = int(eta // 60) - 60 * eta_hours
                    eta_seconds = int(eta % 60)
                    if not eta_hours:
                        logger.info(
                            f"Est time until end of training: {eta_minutes}:{eta_seconds:0>2d}"
                        )
                    else:
                        logger.info(
                            f"Est time until end of training: {eta_hours}:{eta_minutes:0>2d}:{eta_seconds:0>2d}"
                        )

                    logger.info("")
        # Validation period
        epoch += 1
        n_mb_processed_cur_epoch = 0

        validation_targets = []
        validation_predictions = []
        for batch in val_dloader:
            opt.zero_grad()

            audio, labels = batch["audio"], batch["labels"]
            rand_perm = torch.randperm(labels.shape[1])
            audio = audio.cuda()
            labels = labels[:, rand_perm, ...].cuda()
            class_label = rand_perm.numpy().argmin()

            embedded_audio = model.embed_audio(audio)
            embedded_locations = model.embed_location(labels)

            scores = model.score_pairs(embedded_audio, embedded_locations)
            predictions = scores.argmax(dim=1)

            validation_targets.extend([class_label] * audio.shape[0])
            validation_predictions.extend(predictions.detach().cpu().numpy())

        validation_targets = np.array(validation_targets)
        validation_predictions = np.array(validation_predictions)
        accuracy = (validation_targets == validation_predictions).mean()
        logger.info(f"Validation accuracy: {accuracy:.1%}")

        logger.info("Saving state")
        if not save_directory:
            logger.warn("No weights path specified, not saving")
        else:
            wpath = save_directory / f"weights_{epoch}.pt"
            torch.save(model.state_dict(), wpath)
        logger.info("")
    logger.info("Done.")


def eval(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index: Optional[np.ndarray] = None,
) -> None:
    """Runs evaluation on a trained model."""

    # First find the most recent weights file
    weights_files = list(save_directory.glob("weights_*.pt"))
    weights_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    weights_file = weights_files[-1]

    print(f"Loading weights from {weights_file}")
    weights = torch.load(weights_file)
    model: Wavenet = Wavenet(config)
    model.load_state_dict(weights)
    model.cuda()
    model.eval()

    if index is None:
        with h5py.File(data_path, "r") as ctx:
            dset_size = len(ctx["length_idx"]) - 1
            index = np.arange(dset_size)

    dataloader = load_single_dataloader(config, data_path, index)

    all_scores = []
    for batch in tqdm(dataloader):
        audio, labels = batch["audio"], batch["labels"]
        audio = audio.cuda()
        # labels should have shape (batch_size, num_animals, 6)
        labels = labels.cuda()

        if labels.shape[-2] == 1:
            raise ValueError("Only one animal in the dataset, cannot evaluate.")
        if labels.shape[-1] != model.location_input_dims:
            raise ValueError(
                f"Expected {model.location_input_dims} location dimensions, got {labels.shape[-1]}"
            )

        embedded_audio = model.embed_audio(audio)
        embedded_locations = model.embed_location(labels)

        scores = model.score_pairs(embedded_audio, embedded_locations)
        # scores = F.softmax(scores, dim=1)
        all_scores.append(scores.detach().cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    output_fname = data_path.stem + "_scores.npy"
    np.save(save_directory / output_fname, all_scores)


def load_single_dataloader(config: dict, data_path: Path, index: np.ndarray):
    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"]["CROP_LENGTH"]
    normalize_data = config["DATA"].get("NORMALIZE_DATA", True)
    sample_rate = config["DATA"].get("SAMPLE_RATE", 192000)
    node_names = config["DATA"].get("NODES_TO_LOAD", None)

    vocalization_dir = config["DATA"].get(
        "VOCALIZATION_DIR",
        None,
    )

    dataset = VocalizationDataset(
        data_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        inference=True,
        index=index,
        normalize_data=normalize_data,
        sample_rate=sample_rate,
        sample_vocalization_dir=vocalization_dir,
        nodes=node_names,
    )

    try:
        # I think this function only exists on linux
        avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        avail_cpus = max(1, os.cpu_count() - 1)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=avail_cpus,
        shuffle=False,
        collate_fn=dataset.collate,
    )

    return dataloader
