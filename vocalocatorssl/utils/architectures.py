from typing import Literal, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import convolve
from torchaudio.models import Conformer

from .profiling import record
from .resnet1d import ResNet1D


def gradsafe_sum(x_list: list[torch.Tensor]) -> torch.Tensor:
    """Sum a Python list of tensors, ensuring that the gradients are properly propagated

    Args:
        x_list (list[torch.Tensor]): List of tensors to sum

    Raises:
        ValueError: If the list is empty

    Returns:
        torch.Tensor: Sum of the tensors
    """
    if not x_list:
        raise ValueError("x_list must not be empty")
    result = x_list[0]
    for x in x_list[1:]:
        result = result + x
    return result


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


class VocalocatorSimpleLayer(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        filter_size: int,
        *,
        downsample: bool,
        dilation: int,
        use_bn: bool = True,
    ):
        super(VocalocatorSimpleLayer, self).__init__()
        self.fc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            stride=(2 if downsample else 1),
            dilation=dilation,
        )
        self.gc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            stride=(2 if downsample else 1),
            dilation=dilation,
        )
        self.batch_norm = (
            torch.nn.BatchNorm1d(channels_out) if use_bn else nn.Identity()
        )

    def forward(self, x):
        fcx = self.fc(x)
        fcx_activated = torch.tanh(fcx) * 0.95 + fcx * 0.05

        gcx = self.gc(x)
        gcx_activated = torch.sigmoid(gcx)

        prod = fcx_activated * gcx_activated
        return self.batch_norm(prod)


class AudioEmbedder(nn.Module):
    def __init__(self, d_embedding: int, num_channels: int):
        super(AudioEmbedder, self).__init__()

        self.d_embedding = d_embedding
        self.num_channels = num_channels

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Embeds an audio sample into a fixed-size embedding.

        Args:
            x (torch.Tensor): Audio sample of shape (batch, num_channels, time)

        Raises:
            NotImplementedError: Must be implemented by subclasses

        Returns:
            torch.Tensor: An embedding of shape (batch, d_embedding)
        """
        raise NotImplementedError(
            "AudioEmbedder.forward must be implemented by subclasses"
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.embed_audio(audio)


class Wavenet(AudioEmbedder):
    def __init__(
        self,
        d_embedding: int,
        num_channels: int,
        *,
        num_blocks: int = 10,
        conv_channels: int = 64,
        kernel_size: int = 7,
        dilation: int = 3,
        xcorr_pairs: Optional[list[tuple[int, int]]] = None,
        xcorr_length: int = 256,
        xcorr_hidden: int = 512,
    ):
        super(Wavenet, self).__init__(d_embedding, num_channels)

        self.num_blocks = num_blocks
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.xcorr_pairs = xcorr_pairs
        self.xcorr_length = xcorr_length
        self.xcorr_hidden = xcorr_hidden
        self.xcorr_mlp = None

        if xcorr_pairs is not None:
            self.xcorr_mlp = nn.Sequential(
                nn.BatchNorm1d(len(xcorr_pairs) * xcorr_length),
                nn.Linear(len(xcorr_pairs) * xcorr_length, xcorr_hidden),
                nn.ReLU(),
                nn.Linear(xcorr_hidden, xcorr_hidden),
                nn.ReLU(),
            )

        self.blocks = nn.ModuleList(
            [
                WavenetBlock(
                    in_channels=conv_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    dilation=i % 4 + 1,
                    injection_channels=None if self.xcorr_mlp is None else xcorr_hidden,
                )
                for i in range(num_blocks)
            ]
        )
        self.initial_conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding="same",
        )
        self.final_dense = nn.Linear(conv_channels, d_embedding)

    def make_xcorrs(self, audio: torch.Tensor) -> torch.Tensor:
        """Computes cross-correlations between the configured channel pairs

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            torch.Tensor: Tensor of xcorrs. Shape (batch, num_pairs * xcorr_length)

        Raises:
            ValueError: If no cross-correlation pairs are configured
        """
        if self.xcorr_pairs is None:
            raise ValueError("No cross-correlation pairs configured")
        xcorrs = []
        x_len = audio.shape[-1]
        xcorr_len = self.xcorr_length
        centered_x = audio[
            ..., x_len // 2 - xcorr_len // 2 : x_len // 2 + xcorr_len // 2
        ]
        for i, j in self.xcorr_pairs:
            a = centered_x[:, i]
            # Use flip to compute cross-correlation instead of convolution
            b = torch.flip(centered_x[:, j], [-1])
            xcorr = convolve(a, b, mode="same")
            xcorrs.append(xcorr)

        return torch.cat(xcorrs, dim=-1)

    @record
    def embed_audio(self, audio: torch.Tensor):
        """Embeds multi-channel audio into a fixed-size representation


        Args:
            audio: (*batch, time, channels) tensor of audio waveforms
        """
        audio = audio.transpose(
            -1, -2
        )  # transpose to (batch, channels, time) required by conv1d
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
        # Output shape is (*batch, d_conv)
        return self.final_dense(obo_sum)

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)


class SimpleNet(AudioEmbedder):
    def __init__(
        self,
        d_embedding: int,
        num_channels: int,
        *,
        use_batchnorm: bool = True,
        should_downsample: list[bool] = [False, True] * 5,
        filter_sizes: list[int] = [33] * 10,
        conv_channels: list[int] = [16, 16, 32, 32, 64, 64, 128, 128, 256, 256],
        dilations: list[int] = [1] * 10,
    ):
        """Simple network based on a gated tanh nonlinearity.

        Args:
            d_embedding (int): Size of spatail audio embedding
            num_channels (int): Number of audio channels in input
            use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
            should_downsample (list[bool], optional): Whether to downsample at each layer. Defaults to alternating False and True
            filter_sizes (list[int], optional): Size of convolutional filters. Defaults to 33 for all layers.
            conv_channels (list[int], optional): Number of channels in each convolutional layer. Defaults to [16, 16, 32, 32, 64, 64, 128, 128, 256, 256].
            dilations (list[int], optional): Dilation of each convolutional layer. Defaults to 1.

        Raises:
            ValueError: If the lengths of filter_sizes, conv_channels, dilations, and should_downsample are not the same
        """
        super(SimpleNet, self).__init__(d_embedding, num_channels)
        self.use_batchnorm = use_batchnorm
        self.should_downsample = should_downsample
        self.filter_sizes = filter_sizes
        self.conv_channels = conv_channels
        self.dilations = dilations

        if not (
            (len(self.filter_sizes) == len(self.conv_channels))
            and (len(self.conv_channels) == len(self.dilations))
            and (len(self.dilations) == len(self.should_downsample))
        ):
            raise ValueError(
                "Length of filter_sizes, conv_channels, dilations, and should_downsample must be the same"
            )

        conv_channels.insert(0, num_channels)

        convolutions = [
            VocalocatorSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=use_batchnorm,
            )
            for in_channels, out_channels, filter_size, downsample, dilation in zip(
                conv_channels[:-1],
                conv_channels[1:],
                filter_sizes,
                should_downsample,
                dilations,
            )
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        self.final_pooling = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(conv_channels[-1], d_embedding)

    @record
    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.transpose(
            -1, -2
        )  # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        h1 = self.conv_layers(audio)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        embedding = self.final_linear(h2)
        return embedding

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)


class ResnetConformer(AudioEmbedder):
    def __init__(
        self,
        d_embedding: int,
        num_channels: int,
        *,
        resnet_num_blocks: int = 10,
        resnet_conv_channels: int = 64,
        resnet_kernel_size: int = 7,
        conformer_num_layers: int = 12,
        conformer_kernel_size: int = 11,
        conformer_num_heads: int = 4,
        conformer_mlp_dim: int = 512,
    ):
        super(ResnetConformer, self).__init__(d_embedding, num_channels)
        self.resnet_conv_channels = resnet_conv_channels
        self.resnet_num_blocks = resnet_num_blocks
        self.resnet_kernel_size = resnet_kernel_size
        self.conformer_num_layers = conformer_num_layers
        self.conformer_kernel_size = conformer_kernel_size
        self.conformer_num_heads = conformer_num_heads
        self.conformer_mlp_dim = conformer_mlp_dim

        self.resnet = ResNet1D(
            in_channels=num_channels,
            base_filters=resnet_conv_channels // 2 ** (resnet_num_blocks // 8),
            kernel_size=resnet_kernel_size,
            stride=2,
            groups=1,
            n_block=resnet_num_blocks,
            downsample_gap=4,
            increasefilter_gap=8,
        )
        self.conformer = Conformer(
            input_dim=resnet_conv_channels,
            num_heads=conformer_num_heads,
            num_layers=conformer_num_layers,
            ffn_dim=conformer_mlp_dim,
            depthwise_conv_kernel_size=conformer_kernel_size,
        )

        self.dense = nn.Sequential(
            nn.Linear(resnet_conv_channels, 512),
            nn.ReLU(),
            nn.Linear(512, d_embedding),
        )

    @record
    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = torch.transpose(
            audio, -1, -2
        )  # transpose from (batch, time, channels) to (batch, channels, time)

        resnet_output = self.resnet(audio)

        conformer_input = resnet_output.permute(
            0, 2, 1
        )  # (batch, channels, time) -> (batch, time, channels)
        lengths = torch.full(
            (conformer_input.size(0),), conformer_input.size(1), dtype=torch.int64
        ).to(conformer_input.device)
        conformer_output, _ = self.conformer(conformer_input, lengths)
        conformer_output = conformer_output.mean(dim=1)
        output = self.dense(conformer_output)
        return output

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
