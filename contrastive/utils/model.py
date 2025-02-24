from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import convolve


class FourierEmbedding(nn.Module):
    def __init__(self, d_location: int, d_embedding: int, init_bandwidth: float = 1):
        """Creates a learned positional embedding based on Fourier Features.
        The bandwidth is provided in the same units as the locations.
        """
        super().__init__()
        if d_embedding % 2 != 0:
            raise ValueError(f"d_embedding must be even, got {d_embedding}")
        self.d_location = d_location
        self.d_embedding = d_embedding
        self.bandwidth = init_bandwidth

        self.rand_projection = nn.Parameter(
            torch.zeros(d_location, d_embedding // 2), requires_grad=True
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.rand_projection, mean=0, std=1 / self.bandwidth)

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        mapping = torch.einsum("...m,md->...d", locations, self.rand_projection)
        return torch.cat([torch.cos(mapping), torch.sin(mapping)], dim=-1) / np.sqrt(
            self.d_embedding
        )


class WavenetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        kernel_size: int,
        dilation: int,
        bias=True,
        *,
        injection_channels: Optional[int] = None,
    ):
        super().__init__()
        self.injection_linear = None
        if injection_channels is not None:
            self.injection_linear = nn.Linear(injection_channels, in_channels)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_channels * 2,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
            padding="same",
        )
        self.one_by_one = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, injection: Optional[torch.Tensor] = None):
        if self.injection_linear is not None:
            x = x + self.injection_linear(injection)[:, :, None]
        conv_out = self.conv(x)
        tanh, sigmoid = conv_out.chunk(2, dim=-2)
        tanh = torch.tanh(tanh) * 0.95 + 0.05 * tanh
        sigmoid = torch.sigmoid(sigmoid)
        activation = tanh * sigmoid
        one_by_one_output = self.one_by_one(activation)
        return one_by_one_output + x, one_by_one_output


def gradsafe_sum(x_list: list[torch.Tensor]) -> torch.Tensor:
    result = x_list[0]
    for x in x_list[1:]:
        result = result + x
    return result


class Wavenet(nn.Module):
    defaults = {
        "NUM_BLOCKS": 10,
        "CONV_CHANNELS": 64,
        "KERNEL_SIZE": 7,
        "DILATION": 3,
        "XCORR_PAIRS": None,
        "XCORR_LENGTH": 256,
        "XCORR_HIDDEN": 512,
        "NUM_LOCATION_COORDS": 6,
        "LOCATION_EMBEDDING_DIM": 256,
    }

    def __init__(self, config: dict):
        super().__init__()

        N = config["DATA"]["NUM_MICROPHONES"]

        self.config = config
        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = Wavenet.defaults.copy()
        model_config.update(config.get("MODEL_PARAMS", {}))
        config["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        self.xcorr_mlp = None
        if model_config["XCORR_PAIRS"] is not None:
            self.xcorr_pairs = model_config["XCORR_PAIRS"]
            self.xcorr_length = model_config["XCORR_LENGTH"]
            xcorr_hidden = model_config["XCORR_HIDDEN"]

            self.xcorr_mlp = nn.Sequential(
                nn.BatchNorm1d(len(self.xcorr_pairs) * model_config["XCORR_LENGTH"]),
                nn.Linear(
                    len(self.xcorr_pairs) * model_config["XCORR_LENGTH"], xcorr_hidden
                ),
                nn.ReLU(),
                nn.Linear(xcorr_hidden, xcorr_hidden),
                nn.ReLU(),
            )

        self.blocks = nn.ModuleList(
            [
                WavenetBlock(
                    in_channels=model_config["CONV_CHANNELS"],
                    conv_channels=model_config["CONV_CHANNELS"],
                    kernel_size=model_config["KERNEL_SIZE"],
                    dilation=i % 4 + 1,
                    injection_channels=None if self.xcorr_mlp is None else xcorr_hidden,
                )
                for i in range(model_config["NUM_BLOCKS"])
            ]
        )
        self.initial_conv = nn.Conv1d(
            in_channels=N,
            out_channels=model_config["CONV_CHANNELS"],
            kernel_size=model_config["KERNEL_SIZE"],
            stride=1,
            dilation=1,
            padding="same",
        )

        self.location_input_dims = model_config["NUM_LOCATION_COORDS"]
        location_embedding_dims = model_config["CONV_CHANNELS"]
        self.location_embedding = nn.Sequential(
            nn.Linear(
                self.location_input_dims,
                location_embedding_dims,
            ),
            nn.ReLU(),
            nn.Linear(
                location_embedding_dims,
                location_embedding_dims,
            ),
            nn.ReLU(),
        )
        # self.location_embedding = FourierEmbedding(
        #     self.location_input_dims, location_embedding_dims, init_bandwidth=0.01
        # )

        self.scorer = nn.Sequential(
            nn.Linear(model_config["CONV_CHANNELS"], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def make_xcorrs(self, x: torch.Tensor) -> torch.Tensor:
        """Computes cross-correlations between the configured channel pairs

        Args:
            x: Input tensor of shape (batch, channels, time)
        """
        xcorrs = []
        x_len = x.shape[-1]
        xcorr_len = self.xcorr_length
        centered_x = x[..., x_len // 2 - xcorr_len // 2 : x_len // 2 + xcorr_len // 2]
        for i, j in self.config["MODEL_PARAMS"]["XCORR_PAIRS"]:
            a = centered_x[:, i]
            # Use flip to compute cross-correlation instead of convolution
            b = torch.flip(centered_x[:, j], [-1])
            xcorr = convolve(a, b, mode="same")
            xcorrs.append(xcorr)

        return torch.cat(xcorrs, dim=-1)

    def embed_audio(self, audio: torch.Tensor):
        """Embeds multi-channel audio into a fixed-size representation


        Args:
            audio: (*batch, channels, time) tensor of audio waveforms
        """
        audio = torch.einsum("...tc->...ct", audio)  # transpose
        xcorrs = None
        if self.xcorr_mlp is not None:
            xcorrs = self.make_xcorrs(audio)
            xcorrs = self.xcorr_mlp(xcorrs)

        output = self.initial_conv(audio)
        one_by_ones = []
        for block in self.blocks:
            output, obo = block(output, xcorrs)
            one_by_ones.append(obo)
        # Mean over time dim
        obo_sum = gradsafe_sum(one_by_ones).mean(dim=-1) / len(one_by_ones)
        return obo_sum

    def embed_location(self, locations: torch.Tensor):
        """Embeds a frame from an animal's tracks into a fixed-size representation


        Args:
            locations: (*batch, num_coords) tensor of location coordinates
        """
        return self.location_embedding(locations)

    def score_pairs(
        self, audio_embedding: torch.Tensor, location_embeddings: torch.Tensor
    ):
        """Scores an audio representation against many source location candidates

        Args:
            audio_embedding: (batch, embedding_dim) audio representation
            location_embeddings: (batch, num_locations, embedding_dim) location representations
        """
        combined = audio_embedding[:, None, :] + location_embeddings
        scores = self.scorer(combined).reshape(*location_embeddings.shape[:2])
        # return F.softmax(scores, dim=-1)
        return scores  # Should have shape (batch, num_locations)

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
