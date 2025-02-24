from collections import namedtuple
from math import comb
from typing import Callable, List, Tuple

import numpy as np
import torch
from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.outputs import ModelOutputFactory
from torch import nn
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

from .encodings import FixedEncoding


class Exponentiate(nn.Module):
    """Computes exp(x)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class SkipConnection(torch.nn.Module):
    def __init__(self, submodule: nn.Module):
        super(SkipConnection, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class GerbilizerSimpleLayer(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        filter_size: int,
        *,
        downsample: bool,
        dilation: int,
        use_bn: bool = True,
        transposed: bool = False,
        output_padding: int = 0,
    ):
        super(GerbilizerSimpleLayer, self).__init__()

        self.transposed = transposed
        if not transposed:
            self.fc = torch.nn.Conv1d(
                channels_in,
                channels_out,
                filter_size,
                dilation=dilation,
                stride=2 if downsample else 1,
            )
            self.gc = torch.nn.Conv1d(
                channels_in,
                channels_out,
                filter_size,
                dilation=dilation,
                stride=2 if downsample else 1,
            )
        else:
            self.fc = torch.nn.ConvTranspose1d(
                channels_in,
                channels_out,
                filter_size,
                dilation=dilation,
                stride=2 if downsample else 1,
                output_padding=output_padding,
            )
            self.gc = torch.nn.ConvTranspose1d(
                channels_in,
                channels_out,
                filter_size,
                dilation=dilation,
                stride=2 if downsample else 1,
                output_padding=output_padding,
            )

        self.downsample = downsample
        self.one_by_one = torch.nn.Conv1d(channels_in, channels_out, 3, padding="same")
        self.norm_1 = torch.nn.BatchNorm1d(channels_out) if use_bn else nn.Identity()
        self.downsampler = nn.AvgPool1d(2, 2) if downsample else nn.Identity()

    def forward(self, x):
        fcx = self.fc(x)
        gcx = self.gc(x)
        conv = torch.tanh(fcx) * torch.sigmoid(gcx)
        # conv = F.relu(fcx)
        return self.norm_1(conv)


def compute_output_paddings(
    seq_len: int,
    encoder_kernel_sizes: List[int],
    encoder_strides: List[int],
    encoder_dilations: List[int],
    encoder_paddings: List[int],
) -> Tuple[List[int], int]:
    """Computes the necessary output paddings for each layer of the decoder.
    Also computes the necessary length of the decoder's seed sequence.
    """
    # Output length formula from pytorch conv1d documentation:
    # L: input length, P: padding, D: dilation, K: kernel size, S: stride
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    conv1d_out_len = lambda l, p, d, k, s: (l + 2 * p - d * (k - 1) - 1) // s + 1
    # Output length formula from pytorch transposed conv1d documentation:
    # L: input length, P: padding, D: dilation, K: kernel size, S: stride, O: output padding
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    tconv1d_out_len = lambda l, p, d, k, s, o: (l - 1) * s - 2 * p + d * (k - 1) + o + 1

    # Compute lengths after each layer of encoder
    encoder_lengths = [seq_len]
    for k, s, d, p in zip(
        encoder_kernel_sizes, encoder_strides, encoder_dilations, encoder_paddings
    ):
        encoder_lengths.append(conv1d_out_len(encoder_lengths[-1], p, d, k, s))

    # Compute lengths after each layer of decoder
    decoder_output_paddings = []
    for idx in range(len(encoder_lengths) - 1):
        l_in = encoder_lengths[-idx - 1]  # indexes from -1 to 1
        l_out = encoder_lengths[-idx - 2]  # indexes from -2 to 0
        k = encoder_kernel_sizes[-idx - 1]  # indexes from -1 to 0
        s = encoder_strides[-idx - 1]  # indexes from -1 to 0
        d = encoder_dilations[-idx - 1]  # indexes from -1 to 0
        p = encoder_paddings[-idx - 1]  # indexes from -1 to 0

        len_without_o_pad = tconv1d_out_len(l_in, p, d, k, s, 0)
        necessary_o_pad = l_out - len_without_o_pad
        decoder_output_paddings.append(necessary_o_pad)

    return decoder_output_paddings, encoder_lengths[-1]


class GerbilizerVAE(nn.Module):
    defaults = {
        "USE_BATCH_NORM": True,
        "SHOULD_DOWNSAMPLE": [True, True, True, True, True, True, True, True, True],
        "CONV_FILTER_SIZES": [11] * 9,
        "CONV_NUM_CHANNELS": [64, 64, 128, 128, 256, 256, 512, 512, 512],
        "CONV_DILATIONS": [2, 2, 2, 1, 1, 1, 1, 1, 1],
        "OUTPUT_COV": True,
        "REGULARIZE_COV": False,
        "CPS_NUM_LAYERS": 3,
        "CPS_HIDDEN_SIZE": 1024,
        "LATENT_DIMS": 32,
        "LATENT_COV_RANK": 1,
    }

    ReturnType = namedtuple(
        "GerbilizerVAEOutput",
        ["reconstruction", "mu", "cov_diag", "cov_factors"],
    )

    def __init__(self, config: dict):
        super(GerbilizerVAE, self).__init__()

        N = config["DATA"]["NUM_MICROPHONES"]

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = GerbilizerVAE.defaults.copy()
        model_config.update(config.get("MODEL_PARAMS", {}))
        config["MODEL_PARAMS"] = model_config

        self.beta = config["OPTIMIZATION"]["BETA_COEFF"]

        should_downsample = model_config["SHOULD_DOWNSAMPLE"]
        self.n_channels = model_config["CONV_NUM_CHANNELS"]
        filter_sizes = model_config["CONV_FILTER_SIZES"]
        dilations = model_config["CONV_DILATIONS"]

        use_batch_norm = model_config["USE_BATCH_NORM"]

        self.n_channels.insert(0, N)

        encoder_convolutions: List[nn.Module] = [
            GerbilizerSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=use_batch_norm,
            )
            for in_channels, out_channels, filter_size, downsample, dilation in zip(
                self.n_channels[:-1],
                self.n_channels[1:],
                filter_sizes,
                should_downsample,
                dilations,
            )
        ]
        self.encoder_convolutions = nn.ModuleList(encoder_convolutions)

        reversed_channels = list(reversed(self.n_channels))
        reversed_filter_sizes = list(reversed(filter_sizes))
        reversed_downsample = list(reversed(should_downsample))
        reversed_dilations = list(reversed(dilations))
        # hand crafted and hard coded based on my observation of the sequence lengths
        encoder_strides = [2 if d else 1 for d in should_downsample]
        output_padding, self.encoder_end_length = compute_output_paddings(
            config["DATA"]["CROP_LENGTH"],
            filter_sizes,
            encoder_strides,
            dilations,
            [0] * len(filter_sizes),
        )

        # output_padding = [1, 1, 1, 0, 1, 1, 1, 1, 0]

        decoder_convolutions = [
            GerbilizerSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=True,
                transposed=True,
                output_padding=output_padding,
            )
            for in_channels, out_channels, filter_size, downsample, dilation, output_padding in zip(
                reversed_channels[:-1],
                reversed_channels[1:],
                reversed_filter_sizes,
                reversed_downsample,
                reversed_dilations,
                output_padding,
            )
        ]
        self.decoder_convolutions = nn.ModuleList(decoder_convolutions)

        # will be residually connected
        self.final_layer = nn.Conv1d(N, N, 5, padding="same", dilation=1, stride=1)

        # layers for the cps branch of the network:
        cps_initial_channels = comb(N, 2) * 256  # Number of microphone pairs
        cps_num_layers = model_config["CPS_NUM_LAYERS"]
        cps_hidden_size = model_config["CPS_HIDDEN_SIZE"]

        cps_network = [
            nn.Linear(cps_initial_channels, cps_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(cps_num_layers - 1):
            cps_network.append(
                SkipConnection(
                    nn.Sequential(
                        nn.Linear(cps_hidden_size, cps_hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        self.cps_network = nn.Sequential(*cps_network)

        self.encoding_size = self.n_channels[-1] + cps_hidden_size
        self.num_latents = model_config["LATENT_DIMS"]
        self.latent_rank = model_config["LATENT_COV_RANK"]

        self.encoding_to_post_means = nn.Sequential(
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                ),
            ),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                )
            ),
            nn.Linear(self.encoding_size, self.num_latents),
        )
        self.encodings_to_post_cov = nn.Sequential(
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                ),
            ),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                )
            ),
            nn.Linear(self.encoding_size, self.num_latents * self.latent_rank),
        )
        self.encodings_to_post_cov_diag = nn.Sequential(
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                ),
            ),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(
                        self.encoding_size,
                        self.encoding_size,
                    ),
                    nn.ReLU(),
                )
            ),
            nn.Linear(self.encoding_size, self.num_latents),
            Exponentiate(),
        )

        # Layers for the decoder
        decoder_d_model = reversed_channels[0]
        self.decoder_d_model = decoder_d_model
        # First, linear layers to expand the latent space a bit
        self.latent_to_decoding = nn.Sequential(
            nn.Linear(self.num_latents, decoder_d_model),
            nn.ReLU(),
            nn.Linear(decoder_d_model, decoder_d_model * 2),
            nn.ReLU(),
            nn.Linear(
                decoder_d_model * 2, decoder_d_model * self.encoder_end_length // 4
            ),
            nn.ReLU(),
        )
        # expected length of sequence before decoding is 8
        self.latent_to_decoding_convolutional = nn.Conv1d(
            in_channels=decoder_d_model // 4,
            out_channels=decoder_d_model,
            kernel_size=1,
            stride=1,
        )

    def encode(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio, cps = x
        # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        audio = audio.transpose(-1, -2)
        h1 = audio
        for n, conv_layer in enumerate(self.encoder_convolutions):
            h1 = conv_layer(h1)
        h2 = F.adaptive_avg_pool1d(h1, 1).squeeze(-1)

        # cps initial shape (batch, 256, num_channels)
        # goal shape (batch, 256 * num_channels)
        cps = cps.flatten(start_dim=1)
        cps_branch = self.cps_network(cps)
        h2 = torch.cat((h2, cps_branch), dim=-1)

        post_means = self.encoding_to_post_means(h2)
        post_cov_factor = self.encodings_to_post_cov(h2)
        post_cov_diag = self.encodings_to_post_cov_diag(h2)
        # Exponentiation ensures these are positive

        # Reshape cov factor to be (batch, num_latents, cov_rank)
        post_cov_factor = post_cov_factor.view(
            post_cov_factor.shape[0], self.num_latents, self.latent_rank
        )
        return post_means, post_cov_factor, post_cov_diag

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent variable z in R^(num latents) to a multi-channel"""
        # z initial shape: (batch, latent_dim)
        bsz = z.shape[0]

        audio = self.latent_to_decoding(z)  # (batch, decoder_d_model * 2)

        audio = audio.reshape(bsz, -1, self.encoder_end_length)
        audio = self.latent_to_decoding_convolutional(audio)

        for n, conv_block in enumerate(self.decoder_convolutions):
            audio = conv_block(audio)

        # audio should now have shape (batch, num_mics, orig_seq_len)
        audio = torch.tanh(self.final_layer(audio) + audio)
        return audio.transpose(-1, -2)  # For consistence, output with original shape

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
    ) -> ReturnType:
        mu, cov_factor, cov_diag = self.encode(x)
        dist = LowRankMultivariateNormal(mu, cov_factor, cov_diag)
        z = dist.rsample()
        reconstruction = self.decode(z)

        # Initially x[0] has shape (batch, seq_len, channels) but the loss function expects
        # (batch, channels, seq_len)
        orig_audio = x[0].transpose(-1, -2)
        # If this is not made contiguous the loss function throws a rod
        orig_audio = orig_audio.contiguous()

        output_obj = GerbilizerVAE.ReturnType(
            reconstruction=reconstruction,
            mu=mu,
            cov_diag=cov_diag,
            cov_factors=cov_factor,
        )
        return output_obj

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
