import json
import time
from collections import deque
from pathlib import Path
from sys import stderr
from typing import Optional

import h5py
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from . import utils
from .architectures import AudioEmbedder
from .embeddings import LocationEmbedding


def append_loss(
    training_loss_values: list[float],
    validation_loss_values: list[float],
    save_directory: Path,
) -> None:
    """Appends training and validation loss values to an hdf5 file in the save directory

    Args:
        training_loss_values (list[float]): Training loss values to append
        validation_loss_values (list[float]): Validation loss values to append
        save_directory (Path): Model save directory
    """

    loss_path = save_directory / "losses.h5"

    with h5py.File(loss_path, "a") as f:
        if training_loss_values:
            if "training_loss" not in f:
                f.create_dataset("training_loss", data=training_loss_values)
            else:
                old_losses = f["training_loss"][:]
                del f["training_loss"]
                f["training_loss"] = np.concatenate([old_losses, training_loss_values])
        if validation_loss_values:
            if "validation_loss" not in f:
                f.create_dataset("validation_loss", data=validation_loss_values)
            else:
                old_losses = f["validation_loss"][:]
                del f["validation_loss"]
                f["validation_loss"] = np.concatenate(
                    [old_losses, validation_loss_values]
                )


def training_loop(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_dir: Optional[Path] = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available!")

    # Get logger
    save_directory.mkdir(exist_ok=True, parents=True)
    log_path = save_directory / "train_log.log"
    logger = utils.initialize_logger(log_path)

    default_cfg = utils.get_default_config()
    config = utils.update_recursively(config, default_cfg)

    audio_embedder: AudioEmbedder = utils.initialize_audio_embedder(config)
    audio_embedder.cuda()
    location_embedder: LocationEmbedding = utils.initialize_location_embedding(config)
    location_embedder.cuda()
    scorer = utils.initialize_scorer(config)
    scorer.cuda()

    param_list = [
        {"params": audio_embedder.parameters()},
        {"params": location_embedder.parameters()},
        {"params": scorer.parameters()},
    ]
    opt = utils.initialize_optimizer(config, param_list)

    augmentor = utils.initialize_augmentations(config)
    # loss_fn = utils.initialize_loss_function(config)
    train_dloader, val_dloader, test_dloader = utils.initialize_dataloaders(
        config, data_path, index_path=index_dir
    )

    if save_directory is not None:
        with open(save_directory / "config.json", "w") as ctx:
            json.dump(config, ctx, indent=4)
        index_dir = save_directory / "indices"
        index_dir.mkdir(exist_ok=True)
        np.save(index_dir / "train_idx.npy", train_dloader.dataset.index)
        np.save(index_dir / "val_idx.npy", val_dloader.dataset.index)
        if test_dloader is not None:
            np.save(index_dir / "test_idx.npy", test_dloader.dataset.index)

    total_num_updates = config["optimization"]["num_weight_updates"]

    n_warmup_steps = config["optimization"]["num_warmup_steps"]
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
    log_interval = 10

    minibatch_proc_times = deque(maxlen=50)

    epoch = 0
    mb_per_epoch = config["optimization"]["weight_updates_per_epoch"]
    n_mb_processed_cur_epoch = 0
    n_mb_procd = lambda: epoch * mb_per_epoch + n_mb_processed_cur_epoch
    while n_mb_procd() < total_num_updates:
        while n_mb_processed_cur_epoch < mb_per_epoch:
            # Training period
            for batch in train_dloader:
                # Batch is a dictionary with keys 'audio' and 'labels'
                # audio has shape (batch, time, channels)
                # labels has shape (batch, 1 + num_negative, n_mice, n_nodes, n_dims)
                if n_mb_processed_cur_epoch >= mb_per_epoch:
                    break
                opt.zero_grad()

                audio, labels = batch["audio"], batch["labels"]
                audio = audio.cuda()
                labels = labels.cuda()
                aug_audio = augmentor(audio)

                # Shuffle the positive and negative labels
                positive_label_idx = torch.randint(
                    0, labels.shape[1], (labels.shape[0],)
                ).cuda()
                # Swap the current positive label (at 0) with the randomly selected index
                cur_negatives = labels[range(labels.shape[0]), positive_label_idx, ...]
                labels[range(labels.shape[0]), positive_label_idx, ...] = labels[:, 0]
                labels[:, 0] = cur_negatives

                # embeded_audio: (batch, features)
                embedded_audio: torch.Tensor = audio_embedder(aug_audio)
                # embedded_locations: (batch, 1 + num_negative, features)
                embedded_locations: torch.Tensor = location_embedder(labels)

                # Make audio embeddings broadcastable
                embedded_audio = embedded_audio.unsqueeze(1).expand_as(
                    embedded_locations
                )
                # scores: (batch, 1 + num_negative)
                scores = scorer(embedded_audio, embedded_locations)
                loss = F.cross_entropy(
                    scores,
                    positive_label_idx,
                    reduction="none",
                )  # (batch,)

                loss.mean().backward()
                if config["optimization"]["clip_gradients"]:
                    try:
                        audio_embedder.clip_grads()
                    except AttributeError:
                        print(
                            "Failed to clip audio embedder grads. Method not implemented",
                            file=stderr,
                        )
                        config["optimization"]["clip_gradients"] = False

                minibatch_proc_times.append(time.time())

                opt.step()
                sched.step()
                # train_dloader.dataset.step_difficulty()
                n_mb_processed_cur_epoch += 1
                # Log progress
                if (n_mb_procd() % log_interval) == 0:
                    logger.info(utils.get_mem_usage())
                    processing_speed = len(minibatch_proc_times) / (
                        minibatch_proc_times[-1] - minibatch_proc_times[0]
                    )
                    logger.info(
                        f"Progress: {n_mb_procd()} / {total_num_updates} weight updates."
                    )
                    # logger.info(
                    #     f"Current sampling difficulty: {train_dloader.dataset.difficulty:.2f} / 1.0"
                    # )
                    # Losses are in the order time, freq, kl, features, discriminator, adversarial
                    loss = loss.detach().mean().cpu().item()
                    append_loss([loss], [], save_directory)
                    logger.info(f"Loss: {loss:.2f}")
                    logger.info(
                        f"Speed: {processing_speed * audio.shape[0]:.1f} vocalizations per second"
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

        validation_predictions = []
        with torch.no_grad():
            for batch in val_dloader:
                audio, labels = batch["audio"], batch["labels"]
                audio = audio.cuda()
                labels = (
                    labels.cuda()
                )  # (batch, 1 + num_negative, n_mice, n_nodes, n_dims)
                aug_audio = augmentor(audio)

                # Shuffle the position of the ground truth label
                ground_truth_index = torch.randint(
                    0, labels.shape[1], (labels.shape[0],)
                ).cuda()
                # Swap the ground truth label with the first label
                cur_negative = labels[range(labels.shape[0]), ground_truth_index, ...]
                labels[range(labels.shape[0]), ground_truth_index, ...] = labels[:, 0]
                labels[:, 0] = cur_negative

                # embeded_audio: (batch, features)
                embedded_audio: torch.Tensor = audio_embedder(aug_audio)
                # embedded_locations: (batch, 1 + num_negative, features)
                embedded_locations: torch.Tensor = location_embedder(labels)

                # Make audio embeddings broadcastable
                embedded_audio = embedded_audio.unsqueeze(1).expand_as(
                    embedded_locations
                )
                # scores: (batch, 1 + num_negative)
                scores = scorer(embedded_audio, embedded_locations)
                predictions = scores.argmax(dim=1)
                validation_predictions.extend(
                    (predictions == ground_truth_index).cpu().numpy()
                )

        validation_predictions = np.array(validation_predictions)
        accuracy = validation_predictions.mean()
        append_loss([], [accuracy], save_directory)
        logger.info(f"Validation accuracy: {accuracy:.1%}")

        logger.info("Saving state")
        if not save_directory:
            logger.warn("No weights path specified, not saving")
        else:
            a_wpath = save_directory / f"aembed_weights.pt"
            l_wpath = save_directory / f"lembed_weights.pt"
            s_wpath = save_directory / f"scorer_weights.pt"
            torch.save(audio_embedder.state_dict(), a_wpath)
            torch.save(location_embedder.state_dict(), l_wpath)
            torch.save(scorer.state_dict(), s_wpath)
            logger.info(f"Saved weights to {a_wpath}, {l_wpath}, {s_wpath}")
        logger.info("")
    logger.info("Done.")


def eval(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_path: Optional[Path] = None,
) -> None:
    """Runs inference using a trained model. For each vocalization and location pair in the inference dataset,
    this produces a distribution over scores for each of the candidate locations by contrasting them with
    randomly sampled locations from other frames in the dataset.

    Args:
        config (dict): Model config
        data_path (Path): Path to dataset
        save_directory (Path): Directory containing model weights
        index (Optional[np.ndarray], optional): Index of samples
    to evaluate. Defaults to None, in which case, all vocalizations are processed
    """
    num_samples_per_vocalization = 1000

    # Check existence of weights
    if not (save_directory / "aembed_weights.pt").exists():
        raise FileNotFoundError("Audio embedder weights not found")
    if not (save_directory / "lembed_weights.pt").exists():
        raise FileNotFoundError("Location embedder weights not found")
    if not (save_directory / "scorer_weights.pt").exists():
        raise FileNotFoundError("Scorer weights not found")
    if index_path is not None and not index_path.exists():
        raise FileNotFoundError("Index file not found")

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available!")

    default_cfg = utils.get_default_config()
    config = utils.update_recursively(config, default_cfg)

    audio_embedder: AudioEmbedder = utils.initialize_audio_embedder(config)
    audio_embedder.cuda()
    location_embedder: LocationEmbedding = utils.initialize_location_embedding(config)
    location_embedder.cuda()
    scorer = utils.initialize_scorer(config)
    scorer.cuda()

    audio_embedder.load_state_dict(
        torch.load(save_directory / "aembed_weights.pt", weights_only=True), strict=True
    )
    location_embedder.load_state_dict(
        torch.load(save_directory / "lembed_weights.pt", weights_only=True), strict=True
    )
    scorer.load_state_dict(
        torch.load(save_directory / "scorer_weights.pt", weights_only=True), strict=True
    )

    dset = utils.initialize_inference_dataset(config, data_path, index_path)
    _, n_mice, n_nodes, n_dims = dset[0][1].shape

    output_scores = np.empty(
        (len(dset), n_mice, num_samples_per_vocalization), dtype=np.float32
    )

    all_saved_locations = []

    with torch.no_grad():
        for idx in tqdm(range(len(dset))):
            audio, location = dset[idx]
            audio = audio.cuda().unsqueeze(0)  # (1, time, channels)
            location = location.cuda()  # (1, n_mice, n_nodes, n_dims)
            audio_embedding = audio_embedder(audio)  # (1, features)

            saved_locations = []
            for mouse_idx in range(2):
                fake_location_batch = (
                    dset.sample_rand_locations(num_samples_per_vocalization)
                    .unsqueeze(1)
                    .cuda()
                )  # (num_samples, 1, n_nodes, n_dims)
                saved_locations.append(fake_location_batch.cpu().numpy())
                lembed_input = (
                    location[:, mouse_idx, :, :]
                    .unsqueeze(1)
                    .expand_as(fake_location_batch)
                )
                lembed_input = torch.cat(
                    [lembed_input, fake_location_batch], dim=1
                )  # (num_samples, 2, n_nodes, n_dims)
                lembeddings = location_embedder(lembed_input)  # (num_samples, features)
                scores = scorer(
                    audio_embedding.expand_as(lembeddings), lembeddings
                )  # (num_samples,)
                output_scores[idx, mouse_idx, :] = scores.cpu().numpy()
            all_saved_locations.append(np.array(saved_locations))

    np.save(save_directory / "locs.npy", np.array(all_saved_locations))
    np.save(save_directory / "output_scores.npy", output_scores)
