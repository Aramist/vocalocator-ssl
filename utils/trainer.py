import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from .dataloaders import build_dataloader
from .gvae import GerbilizerVAE
from .losses import (
    LossHandler,
    MultiScaleDiscriminators,
    frequency_domain_multi_scale_loss,
    time_domain_reconstruction_loss,
)

DEFAULT_CONFIG = {
    "OPTIMIZATION": {
        "NUM_WEIGHT_UPDATES": 80_000,
        "NUM_WARMUP_STEPS": 200,
        "INIT_LR": 0.02,
        "FINAL_LR": 0.00,
        "CLIP_GRADIENTS": True,
        "BETA_COEFF": 0.0,
        "LOSS_WEIGHTS": {
            "DISCRIMINATOR_UPDATE_PROB": 0.4,
            "EXP_AVG_BETA": 0.999,
            "TIME_WEIGHT": 2,
            "FREQ_WEIGHT": 0.2,
            "FREQ_LOSS_NFFTS": [64, 128, 256, 512, 1024, 2048],
            "KL_WEIGHT": 1,
            "FEATURE_MATCHING_WEIGHT": 1e-7,
            "DISCRIMINATOR_WEIGHT": 3,
            "ADVERSARIAL_WEIGHT": 1,
        },
    },
    "DISCRIMINATOR": {
        "NFFTs": [64, 128, 256, 512],
    },
    "DATA": {
        "NUM_MICROPHONES": 4,
        "SAMPLE_RATE": 125000,
        "CROP_LENGTH": 8192,
        "BATCH_SIZE": 400,
        "ARENA_DIMS": [556.9, 355.6],
        "ARENA_DIMS_UNITS": "MM",
        "MAKE_CPS": True,
    },
    "MODEL_PARAMS": {},
    "AUGMENTATIONS": {
        "AUGMENT_DATA": False,
        # Data augmentations: involves performing augmentations to the audio to which the model should be invariant
        "NOISE": {
            "MIN_SNR": 5,
            "MAX_SNR": 15,
            "PROB": 1.0,
        },
        "MASK": {
            "PROB": 0.25,
            "MIN_LENGTH": 75,  # 0.6 ms at 125 kHz
            "MAX_LENGTH": 125,  # 1 ms at 125 kHz
        },
    },
}


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def get_mem_usage():
    if not torch.cuda.is_available():
        return ""
    used_gb = torch.cuda.max_memory_allocated() / (2**30)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
    torch.cuda.reset_peak_memory_stats()
    return "Max mem. usage: {:.2f}/{:.2f}GiB".format(used_gb, total_gb)


def train(config: dict, data_path: Path, save_directory: Optional[Path] = None):
    if not save_directory:
        print("No save directory specified, not saving model checkpoints.")
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available!")

    cfg = DEFAULT_CONFIG.copy()
    config = update_recursively(config, cfg)

    dloader = build_dataloader(data_path, config)

    # Make generator
    vae: nn.Module = GerbilizerVAE(config).cuda()
    # Make discriminator
    discriminators = MultiScaleDiscriminators(
        num_mics=config["DATA"]["NUM_MICROPHONES"],
        nffts=config["DISCRIMINATOR"]["NFFTs"],
    ).cuda()

    loss_computer = LossHandler(
        hyperparams=config["OPTIMIZATION"]["LOSS_WEIGHTS"],
        discriminators=discriminators,
        balance_losses=False,
    ).cuda()

    opt_config = config["OPTIMIZATION"]
    total_num_updates = opt_config["NUM_WEIGHT_UPDATES"]

    opt = optim.SGD(
        [{"params": vae.parameters()}, {"params": discriminators.parameters()}],
        lr=opt_config["INIT_LR"],
        momentum=0.7,
    )

    n_warmup_steps = opt_config["NUM_WARMUP_STEPS"]
    warmup_sched = optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, n_warmup_steps)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=(total_num_updates - n_warmup_steps),
        eta_min=opt_config["FINAL_LR"],
        verbose=False,
    )
    sched = optim.lr_scheduler.SequentialLR(
        opt, [warmup_sched, cosine_sched], milestones=[n_warmup_steps]
    )
    log_interval = 40

    proc_times = deque(maxlen=30)

    n_mb_processed = 0
    while n_mb_processed < total_num_updates:
        for batch in dloader:
            if n_mb_processed > total_num_updates:
                break
            opt.zero_grad()

            audio, cps = batch
            audio = audio.cuda()
            cps = cps.cuda()

            model_output: GerbilizerVAE.ReturnType = vae((audio, cps))

            loss, losses = loss_computer.get_loss(
                orig_audio=audio, model_output=model_output
            )
            loss.backward()

            proc_times.append(time.time())

            opt.step()
            sched.step()
            n_mb_processed += 1
            # Log progress
            if (n_mb_processed % log_interval) == 0:
                print(get_mem_usage())
                processing_speed = len(proc_times) / (proc_times[-1] - proc_times[0])
                print(
                    f"Progress: {n_mb_processed} / {total_num_updates} weight updates."
                )
                # Losses are in the order time, freq, kl, features, discriminator, adversarial
                losses = [l.detach().cpu().item() for l in losses]
                print(f"Losses:\ttime\tfreq\tkl\tfeat      \tdisc\tadv")
                print(
                    f"Losses:\t{losses[0]:.2f}\t{losses[1]:.2f}\t{losses[2]:.2f}\t{losses[3]:.2f}\t{losses[4]:.2f}\t{losses[5]:.2f}"
                )
                print(
                    f"Speed: {processing_speed*audio.shape[0]:.1f} vocalizations per second"
                )
                eta = (total_num_updates - n_mb_processed) / processing_speed
                eta_hours = int(eta // 3600)
                eta_minutes = int(eta // 60) - 60 * eta_hours
                eta_seconds = int(eta % 60)
                if not eta_hours:
                    print(
                        f"Est time until end of training: {eta_minutes}:{eta_seconds:0>2d}"
                    )
                else:
                    print(
                        f"Est time until end of training: {eta_hours}:{eta_minutes:0>2d}:{eta_seconds:0>2d}"
                    )
                print()
                print()

                print("Saving sample output to sample_output.npy")
                sample_output = model_output.reconstruction.cpu().detach().numpy()
                np.save("sample_input.npy", audio.cpu().detach().numpy()[:8, ...])
                np.save("sample_output.npy", sample_output[:8, ...])

            if n_mb_processed % 5_000 == 0:
                print("Saving state")
                if not save_directory:
                    print("No weights path specified, not saving")
                else:
                    wpath = save_directory / f"weights_{n_mb_processed}.pt"
                    torch.save(vae.state_dict(), wpath)
    print("Done.")
