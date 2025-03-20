import typing as tp
from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader

from .architectures import AudioEmbedder, ResnetConformer, SimpleNet, Wavenet
from .augmentations import AugmentationConfig, build_augmentations
from .dataloaders import VocalizationDataset, build_dataloaders, build_inference_dataset
from .embeddings import FourierEmbedding, LocationEmbedding, MLPEmbedding
from .scorers import CosineSimilarityScorer, MLPScorer, Scorer


def get_default_config() -> dict:
    # Keeping this out of the global namespace
    DEFAULT_CONFIG = {
        "optimization": {
            "num_weight_updates": 500_000,
            "weight_updates_per_epoch": 10_000,
            "num_warmup_steps": 10_000,
            "optimizer": "sgd",
            "momentum": 0.7,
            "weight_decay": 0,
            "clip_gradients": True,
            "initial_learning_rate": 0.003,
        },
        # Valid architectures: wavenet, simplenet, conformer
        "architecture": "simplenet",
        "model_params": {
            "d_embedding": 128,
        },
        # Valid location embedding types: fourier, mlp
        "location_embedding_type": "fourier",
        "location_embedding_params": {
            "n_expected_locations": 2,  # dyad by default
            "d_embedding": 128,
            "init_bandwidth": 0.1,
            "location_combine_mode": "concat",
        },
        # Valid scorers: cosinesim, mlp
        "score_function_type": "cosinesim",
        "score_function_params": {},
        # Valid loss functions: crossentropy
        "loss_function": "crossentropy",
        "dataloader": {
            "num_microphones": 24,
            "crop_length": 8192,
            "arena_dims": [615, 615, 425],
            "arena_dims_units": "mm",
            "normalize_data": True,
            "nodes_to_load": ["Nose", "Head"],
            "batch_size": 128,
            "num_negative_samples": 1,
            "num_inference_samples": 1000,
            # "min_difficulty": 300,
            # "max_difficulty": 30,
            # "sample_rate": 250000,
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
        "inference": {
            "num_samples_per_vocalization": 1000,
        },
    }
    return DEFAULT_CONFIG


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def initialize_optimizer(
    config: dict, params: list | dict | tp.Iterator[nn.Parameter]
) -> optim.Optimizer:
    """Initializes an optimizer based on the configuration. References the `optimization/optimizer` key
    to determine the optimizer type and the `optimization/*` keys to determine the optimizer specific
    hyperparameters.

    Args:
        config (dict): Configuration dictionary
        params (list | dict): Parameters to optimize

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
    elif config["optimization"]["optimizer"] == "adam":
        base_class = optim.Adam
        kwargs["betas"] = config["optimization"].get("adam_betas", (0.9, 0.999))
    elif config["optimization"]["optimizer"] == "adamw":
        base_class = optim.AdamW
        kwargs["betas"] = config["optimization"].get("adam_betas", (0.9, 0.999))
        kwargs["weight_decay"] = config["optimization"].get("weight_decay", 0)
    else:
        raise ValueError(
            f"Unrecognized optimizer {config['optimization']['optimizer']}. Should be one of 'sgd', 'adam', 'adamw'."
        )

    opt = base_class(
        params,
        lr=config["optimization"]["initial_learning_rate"],
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

    if arch not in ["wavenet", "simplenet", "conformer"]:
        raise ValueError(
            f"Unrecognized architecture {arch}. Should be either 'wavenet', 'simplenet', or 'conformer'."
        )
    if "d_embedding" not in config["model_params"]:
        raise ValueError("No embedding dimension specified in model params.")

    num_channels = config["dataloader"]["num_microphones"]
    params = config["model_params"]
    params["num_channels"] = num_channels

    model: AudioEmbedder
    if arch == "wavenet":
        model = Wavenet(**params)
    elif arch == "simplenet":
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
    else:
        raise ValueError(f"Unrecognized location embedding type {emb_type}")

    return emb


def initialize_dataloaders(
    config: dict, dataset_path: Path, index_path: tp.Optional[Path]
) -> tuple[DataLoader, DataLoader, tp.Optional[DataLoader]]:
    """Initializes the training, validation, and (optionally) test dataloaders.

    Args:
        config (dict): Configuration dictionary
        dataset_path (Path): Path to dataset. Expected to be an HDF5 file
        index_path (tp.Optional[Path]): Path to a directory containing numpy files describing
    the test-train split of the dataset. If none is provided, a split will be automatically
    generated.

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
        normalize_data=config["dataloader"].get("normalize_data", True),
        node_names=config["dataloader"].get("nodes_to_load", None),
        num_negative_samples=config["dataloader"].get("num_negative_samples", 1),
    )

    return train_dloader, val_dloader, test_dloader


def initialize_inference_dataloader(
    config: dict, dataset_path: Path, index_path: tp.Optional[Path]
) -> DataLoader:
    """Initializes the inference dataset

    Args:
        config (dict): Configuration dictionary
        dataset_path (Path): Path to dataset. Expected to be an HDF5 file
        index_path (tp.Optional[Path]): Path to a numpy file containing the indices of the
    dataset to use for inference. If none is provided, the entire dataset will be used.
    """

    dataloader = build_inference_dataset(
        dataset_path,
        index_path,
        arena_dims=config["dataloader"]["arena_dims"],
        crop_length=config["dataloader"]["crop_length"],
        normalize_data=config["dataloader"].get("normalize_data", True),
        node_names=config["dataloader"].get("nodes_to_load", None),
        distribution_sample_size=config["dataloader"]["num_inference_samples"],
    )

    return dataloader
