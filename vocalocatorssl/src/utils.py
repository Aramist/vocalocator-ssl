import typing as tp
from pathlib import Path

import numpy as np
import pyjson5
from torch import nn, optim
from torch.utils.data import DataLoader

from .audio_embed import AudioEmbedder, ResnetConformer, SimpleNet
from .augmentations import AugmentationConfig, build_augmentations
from .dataloaders import build_dataloaders, build_inference_dataset
from .location_embed import (
    FourierEmbedding,
    LocationEmbedding,
    MLPEmbedding,
    PolynomialFourier,
)
from .scorers import CosineSimilarityScorer, MLPScorer, Scorer


def get_default_config() -> dict:
    # Keeping this out of the global namespace
    DEFAULT_CONFIG = {
        "optimization": {
            "num_weight_updates": 500_000,
            "optimizer": "sgd",
            "momentum": 0.7,
            "weight_decay": 0,
            "clip_gradients": True,
            "initial_learning_rate": 0.003,
            "initial_temperature": 1.0,
            "final_temperature": 0.1,
            "num_temperature_steps": 100_000,
        },
        # Valid architectures: simplenet, conformer
        "architecture": "simplenet",
        "model_params": {
            "d_embedding": 128,
        },
        # Valid location embedding types: fourier, mlp
        "location_embedding_type": "fourier",
        "location_embedding_params": {
            "d_embedding": 128,
            "init_bandwidth": 0.1,
            "multinode_strategy": "absolute",
        },
        # Valid scorers: cosinesim, mlp
        "score_function_type": "cosinesim",
        "score_function_params": {},
        "dataloader": {
            "num_microphones": 24,
            "crop_length": 8192,
            "arena_dims": [615, 615, 425],
            "arena_dims_units": "mm",
            "normalize_data": True,
            "nodes_to_load": ["Nose", "Head"],
            "batch_size": 128,
            "num_negative_samples": 1,
            "min_difficulty": -1,
            "max_difficulty": 1,
            "num_difficulty_steps": 1000,
        },
        "augmentations": {
            "should_use_augmentations": True,
            "mask_temporal_prob": 0.5,
            "mask_temporal_min_length": 512,
            "mask_temporal_max_length": 2048,
            "mask_channels_prob": 0.5,
            "mask_channels_min_channels": 1,
            "mask_channels_max_channels": 12,
            "noise_injection_prob": 0.5,
            "noise_injection_snr_min": 5,
            "noise_injection_snr_max": 12,
        },
        "evaluation": {
            "num_samples_per_vocalization": 200,
        },
        "finetune": {
            "initial_learning_rate": 1e-3,  # If None, will use the learning rate from the standard config
            "momentum": 0.9,
            "num_weight_updates": 100_000,  # Max num weight updates
            "method": "none",  # none, lora, last_layers
            "num_last_layers": 1,  # If method is last_layers, how many layers to finetune
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    }
    return DEFAULT_CONFIG


def load_json(path: Path) -> dict:
    with open(path, "rb") as ctx:
        data = pyjson5.load(ctx)
    return data


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def numpy_softmax(x, axis=None, temperature=1.0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def initialize_optimizer(
    config: dict,
    params: list | dict | tp.Iterator[nn.Parameter],
    is_finetuning: bool = False,
) -> optim.Optimizer:
    """Initializes an optimizer based on the configuration. References the `optimization/optimizer` key
    to determine the optimizer type and the `optimization/*` keys to determine the optimizer specific
    hyperparameters.

    Args:
        config (dict): Configuration dictionary
        params (list | dict): Parameters to optimize
        is_finetuning (bool): Whether the model is being finetuned or not. If True, will use the
            finetuning specific hyperparameters

    Raises:
        ValueError: If the optimizer type is not recognized

    Returns:
        torch.optim.Optimizer: Optimizer
    """

    base_class: tp.Type[optim.Optimizer]
    kwargs = {}
    if config["optimization"]["optimizer"] == "sgd":
        base_class = optim.SGD
        kwargs["momentum"] = config["optimization"].get("momentum", 0.9)
        if is_finetuning:
            kwargs["momentum"] = config["finetune"].get("momentum", kwargs["momentum"])
    elif config["optimization"]["optimizer"] == "adam":
        base_class = optim.Adam
        kwargs["betas"] = config["optimization"].get("adam_betas", (0.9, 0.999))
        if is_finetuning:
            kwargs["betas"] = config["finetune"].get("adam_betas", kwargs["betas"])
    elif config["optimization"]["optimizer"] == "adamw":
        base_class = optim.AdamW
        kwargs["betas"] = config["optimization"].get("adam_betas", (0.9, 0.999))
        kwargs["weight_decay"] = config["optimization"].get("weight_decay", 0)
        if is_finetuning:
            kwargs["betas"] = config["finetune"].get("adam_betas", kwargs["betas"])
            kwargs["weight_decay"] = config["finetune"].get(
                "weight_decay", kwargs["weight_decay"]
            )
    else:
        raise ValueError(
            f"Unrecognized optimizer {config['optimization']['optimizer']}. Should be one of 'sgd', 'adam', 'adamw'."
        )

    learning_rate = config["optimization"]["initial_learning_rate"]
    if is_finetuning:
        learning_rate = config["finetune"].get("initial_learning_rate", learning_rate)

    opt = base_class(
        params,
        lr=learning_rate,
        **kwargs,
    )

    return opt


def initialize_scorer(config: dict) -> Scorer:
    """Initializes the scorer module based on the configuration. This is a module
    which takes in audio and location embeddings and produces a score representing
    how well the audio and location embeddings match.
    References the `score_function_type` key to determine the scorer type and the
    `score_function_params` key to determine the scorer specific hyperparameters.

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If no scorer type is specified
        ValueError: If the scorer type is not recognized

    Returns:
        AffinityScorer: Scorer module
    """

    valid_scorers = ["cosinesim", "mlp"]
    if "score_function_type" not in config:
        raise ValueError("No score function type specified in config.")
    if config["score_function_type"] not in valid_scorers:
        raise ValueError(
            f"Unrecognized score function type {config['score_function_type']}. Should be one of {valid_scorers}"
        )

    d_audio_embed = config["model_params"]["d_embedding"]
    d_location_embed = config["location_embedding_params"]["d_embedding"]
    if config["score_function_type"] == "cosinesim":
        scorer = CosineSimilarityScorer(d_audio_embed)
    else:
        scorer = MLPScorer(
            d_audio_embed, d_location_embed, **config["score_function_params"]
        )

    return scorer


def initialize_augmentations(config: dict) -> nn.Module:
    """Initializes the augmentation module based on the configuration.
    References the `augmentations` key to determine the augmentation type and the
    `augmentation_params` key to determine the augmentation specific hyperparameters.

    Args:
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Augmentation module
    """
    return build_augmentations(AugmentationConfig(**config["augmentations"]))


def initialize_audio_embedder(config: dict) -> AudioEmbedder:
    """Instantiates a neural network based on the configuration.
    References the `architecture` key to determine the model type and the
    `model_params` key to determine the architecture specific hyperparameters.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If no architecture is specified
        ValueError: If no values for d_embed (embedding dimension) are specified

    Returns:
        nn.Module: Model
    """
    arch = config.get("architecture", None)
    if arch is None:
        raise ValueError("No architecture specified in config.")
    arch = arch.lower()

    if arch not in ["simplenet", "conformer"]:
        raise ValueError(
            f"Unrecognized architecture {arch}. Should be either 'simplenet' or 'conformer'."
        )
    if "d_embedding" not in config["model_params"]:
        raise ValueError("No embedding dimension specified in model params.")

    num_channels = config["dataloader"]["num_microphones"]
    params = config["model_params"]
    params["num_channels"] = num_channels

    model: AudioEmbedder
    if arch == "simplenet":
        model = SimpleNet(**params)
    else:
        model = ResnetConformer(**params)

    return model


def initialize_location_embedding(config: dict) -> LocationEmbedding:
    """Instantiates a location embedding based on the configuration.
    References the `location_embedding_type` key to determine the embedding type and the
    `location_embedding_params` key to determine the embedding specific hyperparameters.

    Args:
        config (dict): Config dictionary

    Raises:
        ValueError: If no location embedding type is specified
        ValueError: If the location embedding type is not recognized

    Returns:
        LocationEmbedding: Location embedding
    """

    emb_type = config.get("location_embedding_type", None)
    if emb_type is None:
        raise ValueError("No location embedding type specified in config.")
    emb_type = emb_type.lower()

    params = config.get("location_embedding_params", {})

    params["d_location"] = len(config["dataloader"]["arena_dims"]) * len(
        config["dataloader"]["nodes_to_load"]
    )

    if emb_type == "fourier":
        emb = FourierEmbedding(**config["location_embedding_params"])
    elif emb_type == "mlp":
        emb = MLPEmbedding(**config["location_embedding_params"])
    elif emb_type == "poly-fourier":
        emb = PolynomialFourier(**config["location_embedding_params"])
    else:
        raise ValueError(f"Unrecognized location embedding type {emb_type}")

    return emb


def initialize_dataloaders(
    config: dict,
    dataset_path: Path,
    index_path: tp.Optional[Path],
    rank: int = 0,
) -> tuple[DataLoader, DataLoader, tp.Optional[DataLoader]]:
    """Initializes the training, validation, and (optionally) test dataloaders.

    Args:
        config (dict): Configuration dictionary
        dataset_path (Path): Path to dataset. Expected to be an HDF5 file
        index_path (tp.Optional[Path]): Path to a directory containing numpy files describing
    the test-train split of the dataset. If none is provided, a split will be automatically
    generated.
        global_rank (int, optional): Global rank of the process. Defaults to 0. Used to seed
    the dataset sampler to avoid redundant data loading across multiple processes.
    This is only relevant for distributed training.

    Raises:
        FileNotFoundError: If the dataset file is not found

    Returns:
        Tuple[DataLoader, DataLoader, tp.Optional[DataLoader]]: Training, validation, and test dataloaders
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

    train_dloader, val_dloader, test_dloader = build_dataloaders(
        dataset_path,
        index_path,
        arena_dims=config["dataloader"]["arena_dims"],
        batch_size=config["dataloader"]["batch_size"],
        crop_length=config["dataloader"]["crop_length"],
        normalize_data=config["dataloader"]["normalize_data"],
        node_names=config["dataloader"]["nodes_to_load"],
        num_negative_samples=config["dataloader"]["num_negative_samples"],
        num_difficulty_steps=config["dataloader"]["num_difficulty_steps"],
        min_difficulty=config["dataloader"]["min_difficulty"],
        max_difficulty=config["dataloader"]["max_difficulty"],
        sampler_seed=rank,
        num_val_negative_samples=config["dataloader"].get(
            "num_val_negative_samples", None
        ),
    )

    return train_dloader, val_dloader, test_dloader


def initialize_inference_dataloader(
    config: dict,
    dataset_path: Path,
    index_path: tp.Optional[Path],
    test_mode: bool = False,
) -> DataLoader:
    """Initializes the inference dataset

    Args:
        config (dict): Configuration dictionary
        dataset_path (Path): Path to dataset. Expected to be an HDF5 file
        index_path (tp.Optional[Path]): Path to a numpy file containing the indices of the
    dataset to use for inference. If none is provided, the entire dataset will be used.
        test_mode (bool, optional): If True, the model's accuracy at varying distances will be
    tested on the provided dataset. Otherwise, the model will be set up to predict sound sources.
    Defaults to False.
    """

    num_inference_samples = (
        config["evaluation"]["num_samples_per_vocalization"] if test_mode else 0
    )
    batch_size = (
        config["evaluation"].get("eval_batch_size", 1)
        if test_mode
        else config["dataloader"]["batch_size"]
    )
    dataloader = build_inference_dataset(
        dataset_path,
        index_path,
        arena_dims=config["dataloader"]["arena_dims"],
        crop_length=config["dataloader"]["crop_length"],
        batch_size=batch_size,
        normalize_data=config["dataloader"].get("normalize_data", True),
        node_names=config["dataloader"].get("nodes_to_load", None),
        num_inference_samples=num_inference_samples,
    )

    return dataloader


def compute_confidence_set(pmf: np.ndarray, conf_level: float = 0.95) -> np.ndarray:
    """Computes the confidence set for a given PMF and confidence level.

    Args:
        pmf (np.ndarray): Probability mass function. Shape doesn't matter, but should be
            unbatched.
        conf_level (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        np.ndarray: Confidence set. Bool array of same shape as pmf.
    """

    if pmf.dtype != np.float64:
        print("Warning: pmf is not of type float64. There may be round-off errors.")
    # sort the pmf and compute the cdf
    flat_pmf = pmf.flatten()
    sorting = np.argsort(-flat_pmf)  # Sort descending
    cdf = np.cumsum(flat_pmf[sorting])
    # find the index of the first element in the cdf that exceeds the confidence level
    # This is the last element in the confidence set
    last_idx = np.searchsorted(cdf, conf_level * flat_pmf.sum())
    # create a mask of the confidence set
    conf_set = np.zeros_like(flat_pmf, dtype=bool)
    conf_set[sorting[: last_idx + 1]] = True
    return conf_set.reshape(pmf.shape)


def point_in_conf_set(
    conf_set: np.ndarray, point: np.ndarray, range: np.ndarray
) -> np.ndarray:
    """Checks if a point is in the confidence set.

    Args:
        conf_set (np.ndarray): Confidence set. (n_angle, n_y, n_x) array of bools.
        point (np.ndarray): Points to check. Should have shape (*batch, n_nodes, n_dims).
        range (np.ndarray): Range of the confidence set. Should have shape (d, 2) where d
            is the number of dimensions. Each row should be [min, max] for each dimension.

    Returns:
        bool array: True if the point is in the confidence set, False otherwise. shape (*batch,)
    """

    # Determine the angle bin
    angle_bins = np.linspace(
        0,
        2 * np.pi,
        conf_set.shape[0] + 1,
        endpoint=True,
    )
    x_bins = np.linspace(
        range[0, 0],
        range[0, 1],
        conf_set.shape[2] + 1,
        endpoint=True,
    )
    y_bins = np.linspace(
        range[1, 0],
        range[1, 1],
        conf_set.shape[1] + 1,
        endpoint=True,
    )
    batch_size = point.shape[:-2]
    point = point.reshape(-1, point.shape[-2], point.shape[-1])
    # Compute the yaw
    head_to_nose = point[:, 0, :2] - point[:, 1, :2]  # (n_pts, 2)
    yaw = np.arctan2(head_to_nose[:, 1], head_to_nose[:, 0]) + np.pi  # (n_pts,)
    angle_idx = np.digitize(yaw, angle_bins) - 1  # (n_pts,)
    y_bin = np.digitize(point[:, 0, 1], y_bins) - 1  # (n_pts,)
    y_bin = np.clip(y_bin, 0, conf_set.shape[1] - 1)
    x_bin = np.digitize(point[:, 0, 0], x_bins) - 1  # (n_pts,)
    x_bin = np.clip(x_bin, 0, conf_set.shape[2] - 1)
    # Check if the point is in the confidence set
    result = conf_set[angle_idx, y_bin, x_bin]  # (n_pts,)
    # Reshape the result to match the batch size
    result = result.reshape(*batch_size)
    return result


def compute_test_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    arena_dims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the accuracy of the model at different distances.

    Args:
        labels (np.ndarray): Labels. Shape (n_samples, n_negative + 1, n_nodes, n_dims)
        scores (np.ndarray): Scores. Shape (n_samples, n_negative + 1)
        arena_dims (np.ndarray): Arena dimensions. Shape (3,)

    Returns:
        tuple[np.ndarray, np.ndarray]: Distances and accuracies
    """
    # Drop the z dimension and the head node
    labels = labels[..., 0, :2]  # (B, 1 + n_negative, 2)
    # Convert to cm
    scale_factor = arena_dims.max() / 2 / 10
    labels *= scale_factor

    positive_labels = labels[:, 0]  # (B, 2)
    positive_scores = scores[:, 0]  # (B, )
    negative_labels = labels[:, 1:]  # (B, n_negative, 2)
    negative_scores = scores[:, 1:]  # (B, n_negative)

    # Distance between each positive label and its corresponding negative labels
    pos_to_neg_dist = np.linalg.norm(
        positive_labels[:, None, :] - negative_labels, axis=-1
    )  # (B, n_negative)
    # If this is a True, the negative label is correctly rejected
    assignment = negative_scores < positive_scores[:, None]  # (B, n_negative).

    pos_to_neg_dist = pos_to_neg_dist.reshape(-1)  # (B * n_negative, )
    assignment = assignment.reshape(-1)  # (B * n_negative, )

    # Bin according to distances
    distance_bins = np.linspace(0, 40, 41, endpoint=True)
    bin_indices = np.digitize(pos_to_neg_dist, distance_bins) - 1  # (B * n_negative, )

    # Compute the accuracy for each bin
    bin_acc = np.zeros((len(distance_bins) - 1), dtype=float)
    for j in range(len(distance_bins) - 1):
        bin_mask = bin_indices == j
        bin_acc[j] = assignment[bin_mask].mean() if np.any(bin_mask) else 0.0

    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    return bin_centers, bin_acc


def compute_test_calibration(
    scores: np.ndarray,
    num_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the calibration curve for the model on a dataset with known speaker identity.

    Args:
        scores (np.ndarray): Array of shape (batch, 1 + num_negative_samples) containing the scores for each sample.
            The first entry is expected to be the score of the positive sample, and the remaining entries are the scores
            of the negative samples for the same vocalization.

    Returns:
        tuple[np.ndarray, np.ndarray]: Confidence bin left edges and the corresponding accuracies
    """
    pos_scores = scores[:, 0]  # (B, )
    neg_scores = scores[:, 1:]  # (B, n_negative)

    num_negative = neg_scores.shape[1]
    pos_scores = pos_scores.repeat(num_negative, axis=0)  # (B * n_negative, )
    neg_scores = neg_scores.reshape(-1)  # (B * n_negative, )

    pos_neg_probs = np.stack([pos_scores, neg_scores], axis=1)  # (B * n_negative, 2)
    pos_neg_probs = numpy_softmax(pos_neg_probs, axis=1)

    Y_hat = pos_neg_probs.argmax(axis=1)
    P_hat = pos_neg_probs.max(axis=1)

    p_bins = np.linspace(0.5, 1.0, num_bins + 1, endpoint=True)
    bin_indices = np.digitize(P_hat, p_bins) - 1  # (B * n_negative, )
    bin_acc = np.zeros((num_bins,), dtype=float)
    for j in range(num_bins):
        bin_mask = bin_indices == j
        bin_acc[j] = (Y_hat[bin_mask] == 0).mean() if np.any(bin_mask) else 0.0

    bin_center = 0.5 * (p_bins[:-1] + p_bins[1:])  # (num_bins, )
    return bin_center, bin_acc
