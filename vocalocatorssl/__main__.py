import argparse
from pathlib import Path

import numpy as np
import pyjson5 as json

from .utils.trainer import eval, training_loop


def load_json(path: Path) -> dict:
    with open(path, "rb") as ctx:
        data = json.load(ctx)
    return data


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
        training_loop(config, args.data, args.save_path, args.index)
