import argparse
import json
import os
import typing as tp
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch import callbacks

from .src import utils as utilsmodule
from .src.lightning_wrappers import LVocalocator
from .src.utils import load_json


def find_latest_checkpoint(save_directory: Path) -> Path:
    """Find the latest checkpoint in the given directory.

    Args:
        save_directory (Path): Path to the directory containing checkpoints.

    Returns:
        Path: Path to the latest checkpoint file.
    """
    checkpoints = list(save_directory.glob("**/*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found at {save_directory}")
    newest_checkpoint = max(checkpoints, key=os.path.getctime)
    return newest_checkpoint


def make_trainer(config: dict, save_directory: Path, **kwargs) -> L.Trainer:
    num_nodes = int(os.getenv("SLURM_NNODES", 1))
    return L.Trainer(
        max_steps=config["optimization"]["num_weight_updates"],
        num_nodes=num_nodes,
        default_root_dir=save_directory,
        log_every_n_steps=50,
        callbacks=[
            # Save the best model based on validation accuracy
            callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                save_last=True,
                verbose=False,
            ),
            # End training if validation accuracy does not improve for 40 epochs
            callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=40),
            # End training if weights explode
            callbacks.EarlyStopping(
                monitor="train_loss", check_finite=True, verbose=False, patience=1000
            ),
            # End training if learning rate gets too low
            # callbacks.EarlyStopping(
            #     monitor="lr-SGD",
            #     mode="min",
            #     patience=100000000,
            #     stopping_threshold=1e-6,
            # ),
            # Log learning rates
            callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        gradient_clip_val=1.0 if config["optimization"]["clip_gradients"] else None,
        # limit_train_batches=10,
        # limit_val_batches=1,
        # max_epochs=1,
        # profiler="advanced",
        **kwargs,
    )


def train_default(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_dir: tp.Optional[Path] = None,
    pretrained_path: tp.Optional[Path] = None,
):
    save_directory.mkdir(exist_ok=True, parents=True)

    default_cfg = utilsmodule.get_default_config()
    config = utilsmodule.update_recursively(config, default_cfg)
    trainer = make_trainer(config, save_directory)

    train_dloader, val_dloader, test_dloader = utilsmodule.initialize_dataloaders(
        config, data_path, index_path=index_dir, rank=trainer.global_rank
    )

    if trainer.global_rank == 0:
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

    if pretrained_path is not None:
        pretrained_ckpt = find_latest_checkpoint(pretrained_path)
        print("Loading pretrained model from", pretrained_ckpt)
        # The finetuned model may have a different config, loading the checkpoint without explicitly
        # passing the new config will use the old config from the pretrained model
        model = LVocalocator.load_from_checkpoint(
            pretrained_ckpt, is_finetuning=True, config=config
        )
    else:
        model = LVocalocator(config)
    trainer.fit(model, train_dloader, val_dloader)

    if trainer.global_rank == 0:
        # Get the best checkpoint path
        best_ckpt = trainer.checkpoint_callback.best_model_path
        best_ckpt = os.path.relpath(best_ckpt, start=save_directory)
        # Symlink to save_directory
        if os.path.islink(save_directory / "best.ckpt"):
            os.remove(save_directory / "best.ckpt")
        if os.path.exists(save_directory / "best.ckpt"):
            os.remove(save_directory / "best.ckpt")
        os.symlink(best_ckpt, save_directory / "best.ckpt")


def inference(
    data_path: Path,
    save_directory: Path,
    *,
    output_path: Path,
    index_file: tp.Optional[Path] = None,
    test_mode: bool = False,
    make_pmfs: bool = False,
):
    """Runs inference on a dataset using the trained model located at `save_directory`.

    Args:
        config (dict): Model configuration.
        data_path (Path): Path to the dataset.
        save_directory (Path): Path to the directory containing the trained model.
        index_file (tp.Optional[Path], optional): Path ta a numpy array containing indices to process. Defaults to None.
        output_path (Path): Path to save the predictions.
        test_mode (bool, optional): If False, the model will be used for predicting sound sources. If true
            the model's accuracy at varying distances will be tested on the provided dataset.

    Raises:
        FileNotFoundError: If the model is not found at `save_directory`.
        FileNotFoundError: If no checkpoints are found at `save_directory`.
        ValueError: If the model configuration is not provided and cannot be found in the checkpoint.
    """
    if not save_directory.exists():
        raise FileNotFoundError(f"Model not found at {save_directory}")

    config_path = save_directory / "config.json"
    config = load_json(config_path)

    newest_checkpoint = find_latest_checkpoint(save_directory)

    try:
        # Attempt with strict first to see if LoRA was used
        model = LVocalocator.load_from_checkpoint(
            newest_checkpoint, strict=True, config=config
        )
    except RuntimeError:
        # Lora was probably used, load without strict to get hparams and then finetunify
        model = LVocalocator.load_from_checkpoint(
            newest_checkpoint, strict=False, config=config, is_finetuning=True
        )
        model.finetunify()
        model.load_state_dict(
            torch.load(newest_checkpoint, map_location="cpu")["state_dict"],
            strict=True,
        )

    dloader = utilsmodule.initialize_inference_dataloader(
        model.config, data_path, index_file, test_mode=test_mode
    )

    trainer = make_trainer(
        config,
        save_directory,
        logger=False,
    )
    make_pmfs = make_pmfs and not test_mode  # Mutually exclusive
    model.flags["predict_test_mode"] = test_mode  # Hack to pass args into predict_step
    model.flags["predict_gen_pmfs"] = make_pmfs
    preds: tp.Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        trainer.predict(
            model,
            dloader,
            return_predictions=True,
            ckpt_path=newest_checkpoint,  # probably unnecessary
        )
    )

    labels = [x[0] for x in preds]
    scores = [x[1] for x in preds]

    labels = torch.cat(labels, dim=0).cpu().numpy()
    scores = torch.cat(scores, dim=0).cpu().numpy()

    if make_pmfs:
        pmfs = [x[2] for x in preds]
        pmfs = torch.cat(pmfs, dim=0).cpu().numpy()
        np.savez(output_path, labels=labels, scores=scores, pmfs=pmfs)
    else:
        np.savez(output_path, labels=labels, scores=scores)

    if test_mode:
        # Compute and report accuracy
        dist_bins, accs = utilsmodule.compute_test_accuracy(
            labels, scores, dloader.dataset.arena_dims
        )

        with open(save_directory / "test_accuracy.txt", "w") as ctx:
            header = "Distance (cm),Accuracy (%)"
            print(header)
            ctx.write(header + "\n")
            for dist, acc in zip(dist_bins, accs):
                line = f"{float(dist):.1f},{float(acc):.3f}"
                print(line)
                ctx.write(line + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--save-path", type=Path)
    ap.add_argument("--finetune-from", type=Path)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--index", type=Path)
    ap.add_argument("--gen-pmfs", action="store_true")
    ap.add_argument("-o", "--output-path", type=Path, default=None)
    args = ap.parse_args()
    if args.config is not None:
        # If we have a config, use it to override defualts / pretrain config
        config = load_json(args.config)
    elif args.predict or args.test:
        config = None
    else:
        # config cannot be none during training
        config = {}  # will be filled with default values

    if args.save_path is None:
        args.save_path = Path(".")

    args.save_path = args.save_path.resolve()

    output_path = (
        args.output_path
        if args.output_path is not None
        else args.save_path / "predictions.npz"
    )
    if args.predict:
        inference(
            args.data,
            args.save_path,
            index_file=args.index,
            output_path=output_path,
            make_pmfs=args.gen_pmfs,
        )
    elif args.test:
        inference(
            args.data,
            args.save_path,
            index_file=args.index,
            output_path=output_path,
            test_mode=True,
        )
    else:
        args.save_path.mkdir(parents=True, exist_ok=True)
        train_default(
            config,
            args.data,
            args.save_path,
            args.index,
            pretrained_path=args.finetune_from,
        )
