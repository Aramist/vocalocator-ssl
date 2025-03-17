import os
from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from tqdm import tqdm

from . import utils
from .architectures import AudioEmbedder
from .embeddings import LocationEmbedding
from .lightning_wrappers import LVocalocator


def training_loop(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_dir: Optional[Path] = None,
):
    save_directory.mkdir(exist_ok=True, parents=True)

    default_cfg = utils.get_default_config()
    config = utils.update_recursively(config, default_cfg)

    train_dloader, val_dloader, _ = utils.initialize_dataloaders(
        config, data_path, index_path=index_dir
    )

    model = LVocalocator(config)
    num_nodes = os.getenv("NUM_NODES", 1)
    num_nodes = int(num_nodes)
    trainer = L.Trainer(
        num_nodes=num_nodes,
        max_steps=config["optimization"]["num_weight_updates"],
        default_root_dir=save_directory,
        # check_val_every_n_epoch=config["optimization"]["weight_updates_per_epoch"],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dloader, val_dloader)


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
