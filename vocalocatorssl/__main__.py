import argparse
import os
import typing as tp
from pathlib import Path

import lightning as L
import pyjson5 as json

from .src import utils as utilsmodule
from .src.lightning_wrappers import LVocalocator
from .src.trainer import eval


def load_json(path: Path) -> dict:
    with open(path, "rb") as ctx:
        data = json.load(ctx)
    return data


def train_default(
    config: dict,
    data_path: Path,
    save_directory: Path,
    index_dir: tp.Optional[Path] = None,
):
    save_directory.mkdir(exist_ok=True, parents=True)

    default_cfg = utilsmodule.get_default_config()
    config = utilsmodule.update_recursively(config, default_cfg)

    train_dloader, val_dloader, _ = utilsmodule.initialize_dataloaders(
        config, data_path, index_path=index_dir
    )

    model = LVocalocator(config)
    trainer = L.Trainer(
        max_steps=config["optimization"]["num_weight_updates"],
        default_root_dir=save_directory,
        # check_val_every_n_epoch=config["optimization"]["weight_updates_per_epoch"],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dloader, val_dloader)


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
    else:
        config = {}  # will be filled with default values

    if args.save_path is not None:
        args.save_path.mkdir(parents=True, exist_ok=True)

    if args.eval:
        eval(config, args.data, args.save_path, args.index)
    else:
        train_default(config, args.data, args.save_path, args.index)
