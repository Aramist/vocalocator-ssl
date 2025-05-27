import torch
from torch import nn
from torchaudio.models import Conformer

from .lora import LORA_MHA, LORA_Conv1d
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

    def LoRAfy(self, lora_rank: int, lora_alpha: float):
        """Applies LoRA to the model to make finetuning more efficient.

        Args:
            lora_rank (int): Rank of the weight update.
            lora_alpha (int): Scaling factor applied to weight update.
        """
        self.fc = LORA_Conv1d(self.fc, lora_rank, lora_alpha)
        self.gc = LORA_Conv1d(self.gc, lora_rank, lora_alpha)

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
        audio = audio.float()
        # transpose from (batch, time, channels) to (batch, channels, time)
        audio = audio.transpose(-1, -2)
        return self.embed_audio(audio)

    def LoRAfy(self, lora_rank: int, lora_alpha: float, lora_dropout: float):
        """Applies LoRA to the model.

        Args:
            lora_rank (int): Rank of the LoRA matrices.
            lora_alpha (int): Scaling factor for the LoRA matrices.
            lora_dropout (float): Dropout rate for the LoRA matrices.
        """
        raise NotImplementedError(
            "AudioEmbedder.LORAfy must be implemented by subclasses"
        )

    def last_layer_finetunify(self, num_layers: int):
        """Freezes all but the last k layers of the model.

        Args:
            num_layers (int): Number of layers to keep trainable. If num_layers is negative,
                the last num_layers + k layers will be trainable.

        Raises:
            ValueError: If num_layers is zero
        """
        raise NotImplementedError(
            "AudioEmbedder.last_layer_finetunify must be implemented by subclasses"
        )


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

    def LoRAfy(self, lora_rank: int, lora_alpha: float, lora_dropout: float):
        """Applies LoRA to the model.

        Args:
            lora_rank (int): Rank of the LoRA matrices.
            lora_alpha (int): Scaling factor for the LoRA matrices.
            lora_dropout (float): Dropout rate for the LoRA matrices.
        """
        for layer in self.conv_layers.children():
            if isinstance(layer, VocalocatorSimpleLayer):
                layer.LoRAfy(lora_rank, lora_alpha)

    def last_layer_finetunify(self, num_layers: int):
        """Freezes all but the last k layers of the model.

        Args:
            num_layers (int): Number of layers to keep trainable. If num_layers is negative,
                the last num_layers + k layers will be trainable.

        Raises:
            ValueError: If num_layers is zero
        """
        if num_layers == 0:
            raise ValueError("num_layers must be non-zero")
        conv_layers = list(self.conv_layers.children())
        if num_layers < 0:
            num_layers += len(conv_layers)

        for layer in conv_layers[:-num_layers]:
            layer.requires_grad_(False)

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        h1 = self.conv_layers(audio)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        embedding = self.final_linear(h2)
        return embedding


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

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
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

    def LoRAfy(self, lora_rank: int, lora_alpha: float, lora_dropout: float):
        """Applies LoRA to the model.

        Args:
            lora_rank (int): Rank of the LoRA matrices.
            lora_alpha (int): Scaling factor for the LoRA matrices.
            lora_dropout (float): Dropout rate for the LoRA matrices.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        for layer in self.conformer.conformer_layers:
            layer.self_attn = LORA_MHA(layer.self_attn, lora_rank, lora_alpha)
            layer.self_attn.requires_grad_(True)

    def last_layer_finetunify(self, num_layers: int):
        """Freezes all but the last k layers of the model.

        Args:
            num_layers (int): Number of layers to keep trainable. If num_layers is negative,
                the last num_layers + k layers will be trainable.

        Raises:
            ValueError: If num_layers is zero
        """
        if num_layers == 0:
            raise ValueError("num_layers must be non-zero")
        conformer_layers = self.conformer.conformer_layers
        if num_layers < 0:
            num_layers += len(conformer_layers)

        self.resnet.requires_grad_(False)
        for layer in conformer_layers[:-num_layers]:
            layer.requires_grad_(False)
