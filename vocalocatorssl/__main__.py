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
from .src.dataloaders import PluralVocalizationDataset
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
            # End training if validation accuracy does not improve
            callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=100),
            # End training if weights explode
            callbacks.EarlyStopping(
                monitor="train_loss", check_finite=True, verbose=False, patience=1000
            ),
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
            np.savez(index_dir / "train_set.npz", **train_dloader.dataset.indices)
            np.savez(index_dir / "val_set.npz", **val_dloader.dataset.indices)
            if test_dloader is not None:
                np.savez(index_dir / "test_set.npz", **test_dloader.dataset.indices)

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
    temperature_adjustment: float = 1.0,
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
        make_pmfs (bool, optional): If True, the model will generate probability mass functions (PMFs) for each prediction.
            Defaults to False.
        temperature_adjustment (float, optional): Temperature adjustment for calibration. Defaults to 1.0.

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
    model.flags["temperature_adjustment"] = temperature_adjustment
    preds: tp.Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        trainer.predict(
            model,
            dloader,
            return_predictions=True,
            ckpt_path=newest_checkpoint,  # probably unnecessary
        )
    )

    labels = [x[0].cpu().numpy() for x in preds]
    scores = [x[1].cpu().numpy() for x in preds]

    dset: PluralVocalizationDataset = dloader.dataset
    filenames = dset.filenames
    dset_lengths = dset.lengths

    # Concatenate predictions and labels within each dataset
    labels_by_dataset = []
    scores_by_dataset = []
    for length in dset_lengths:
        accum_labels = []
        accum_scores = []
        while sum(len(arr) for arr in accum_labels) < length:
            # Get the next batch of labels
            accum_labels.append(labels.pop(0))
            accum_scores.append(scores.pop(0))
        labels_by_dataset.append(np.concatenate(accum_labels, axis=0))
        scores_by_dataset.append(np.concatenate(accum_scores, axis=0))

    labels = {
        f"{dataset_name}-labels": labels
        for dataset_name, labels in zip(filenames, labels_by_dataset)
    }
    scores = {
        f"{dataset_name}-scores": scores
        for dataset_name, scores in zip(filenames, scores_by_dataset)
    }

    if make_pmfs:
        pmfs = [x[2] for x in preds]
        pmfs = torch.cat(pmfs, dim=0).cpu().numpy()  # These all have the same shape
        np.savez(output_path, pmfs=pmfs, **labels, **scores)
    else:
        np.savez(output_path, **labels, **scores)

    if test_mode:
        # Compute and report confidence
        # in test mode the num_animal dimensions is reduced out
        # scores should have shape (N, num_negative + 1)
        scores_concat = np.concatenate(
            [scores[f"{dataset_name}-scores"] for dataset_name in filenames], axis=0
        )
        cal_bins, calibration_curve = utilsmodule.compute_test_calibration(
            scores_concat
        )
        with open(save_directory / "test_calibration.txt", "w") as ctx:
            header = "bin_center,accuracy"
            print(header)
            ctx.write(header + "\n")
            for bin_left, acc in zip(cal_bins, calibration_curve):
                line = f"{float(bin_left):.3f},{float(acc):.3f}"
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
    ap.add_argument(
        "--temp-adjustment",
        type=float,
        default=1.0,
        help="Temperature adjustment for calibration",
    )
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
            temperature_adjustment=args.temp_adjustment,
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
