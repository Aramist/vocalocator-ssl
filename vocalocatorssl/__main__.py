import argparse
import json
import os
import typing as tp
from pathlib import Path

import lightning as L
import numpy as np
import pyjson5
import torch
from lightning.pytorch import callbacks

from .src import utils as utilsmodule
from .src.lightning_wrappers import LVocalocator


def load_json(path: Path) -> dict:
    with open(path, "rb") as ctx:
        data = pyjson5.load(ctx)
    return data


def make_trainer(config: dict, save_directory: Path) -> L.Trainer:
    num_nodes = int(os.getenv("SLURM_NNODES", 1))
    return L.Trainer(
        max_steps=config["optimization"]["num_weight_updates"],
        num_nodes=num_nodes,
        default_root_dir=save_directory,
        log_every_n_steps=1,
        callbacks=[
            # Save the best model based on validation accuracy
            callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                save_last=False,
                verbose=False,
            ),
            # End training if validation accuracy does not improve for 10 epochs
            callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=10),
        ],
        limit_predict_batches=1,
    )


def train_default(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_dir: tp.Optional[Path] = None,
):
    save_directory.mkdir(exist_ok=True, parents=True)

    default_cfg = utilsmodule.get_default_config()
    config = utilsmodule.update_recursively(config, default_cfg)

    train_dloader, val_dloader, test_dloader = utilsmodule.initialize_dataloaders(
        config, data_path, index_path=index_dir
    )

    if index_dir is None:
        index_dir = save_directory / "indices"
        index_dir.mkdir(exist_ok=True)
        np.save(index_dir / "train_set.npy", train_dloader.dataset.index)
        np.save(index_dir / "val_set.npy", val_dloader.dataset.index)
        if test_dloader is not None:
            np.save(index_dir / "test_set.npy", test_dloader.dataset.index)

    # Save config
    with open(save_directory / "config.json", "w") as ctx:
        json.dump(config, ctx, indent=4)

    model = LVocalocator(config)
    trainer = make_trainer(config, save_directory)
    trainer.fit(model, train_dloader, val_dloader)

    if trainer.global_rank == 0:
        # Get the best checkpoint path
        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_ckpt = os.path.relpath(best_ckpt, start=save_directory)
        # Symlink to save_directory
        os.symlink(best_ckpt, save_directory / "best.ckpt")


def inference(
    data_path: Path,
    save_directory: Path,
    index_file: tp.Optional[Path] = None,
    output_path: tp.Optional[Path] = None,
):
    """Runs inference on a dataset using the trained model located at `save_directory`.

    Args:
        config (dict): Model configuration.
        data_path (Path): Path to the dataset.
        save_directory (Path): Path to the directory containing the trained model.
        index_file (tp.Optional[Path], optional): Path ta a numpy array containing indices to process. Defaults to None.
        output_path (tp.Optional[Path], optional): Path to save the predictions. Defaults to predictions.npy.

    Raises:
        FileNotFoundError: If the model is not found at `save_directory`.
        FileNotFoundError: If no checkpoints are found at `save_directory`.
        ValueError: If the model configuration is not provided and cannot be found in the checkpoint.
    """
    if not save_directory.exists():
        raise FileNotFoundError(f"Model not found at {save_directory}")
    if output_path is None:
        output_path = save_directory / "predictions.npy"

    config_path = save_directory / "config.json"
    config = load_json(config_path)

    checkpoints = list(save_directory.glob("**/*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found at {save_directory}")
    newest_checkpoint = max(checkpoints, key=os.path.getctime)

    model = LVocalocator.load_from_checkpoint(newest_checkpoint)
    dloader = utilsmodule.initialize_inference_dataloader(
        model.config, data_path, index_file
    )

    trainer = make_trainer(config, save_directory)
    preds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = trainer.predict(
        model, dloader, ckpt_path=newest_checkpoint, return_predictions=True
    )

    labels = [x[0] for x in preds]
    grid = [x[1] for x in preds]
    scores = [x[2] for x in preds]

    labels = torch.cat(labels, dim=0).cpu().numpy()
    grid = torch.cat(grid, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()
    np.savez(output_path, labels=labels, grid=grid, scores=scores)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--save-path", type=Path)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--index", type=Path)
    args = ap.parse_args()
    if args.config is not None:
        config = load_json(args.config)
    elif args.eval:
        config = None
    else:
        # config cannot be none during training
        config = {}  # will be filled with default values

    if args.save_path is not None:
        args.save_path.mkdir(parents=True, exist_ok=True)

    if args.eval:
        inference(args.data, args.save_path, args.index)
    else:
        train_default(config, args.data, args.save_path, args.index)
