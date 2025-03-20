"""Implements several augmentations useful to processing spatial audio data.

Implemented augmentations:
  - Temporal masking: Replaces a random interval of all channels of the input
  with zeros or white noise.
  - Channel masking: Replaces a random subset of channels of the input with zeros or noise
  - Noise injection: Adds white noise to all channels of the input.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Augmentation(nn.Module):
    def __init__(self, prob: float):
        """Abstract class for spatial audio augmentations.
        Does not mutate the input tensor in place.

        Args:
            prob (float): Probability of applying the augmentation to each member of a batch.

        Raises:
            ValueError: If prob is not in the range [0, 1]
        """
        super(Augmentation, self).__init__()

        if prob < 0 or prob > 1:
            raise ValueError(f"prob must be in the range [0, 1], got {prob}")

        self.prob = prob

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Wraps the apply method with a dice roll. In eval mode, this is a no-op.

        Args:
            audio (torch.Tensor): A batch of audio signals with shape (*batch, samples, channels)

        Returns:
            torch.Tensor: The augmented audio signal. Shape does not change
        """
        if not self.training:
            return audio

        orig_shape = audio.shape
        audio = audio.reshape(-1, *orig_shape[-2:])
        batch_size = audio.shape[0]
        batch_mask = torch.rand(batch_size, device=audio.device) < self.prob
        return self.apply(audio, batch_mask).reshape(orig_shape)

    def apply(self, audio: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """Applies the augmentation to the input audio.

        Args:
            audio (torch.Tensor): Audio tensor with shape (batch, samples, channels)
            batch_mask (torch.Tensor): Mask of shape (batch,) indicating which samples to augment.
            A value of 1 (True) indicates the augmentation should be applied to that sample.

        Returns:
            torch.Tensor: The augmented audio signal. Shape does not change
        """
        raise NotImplementedError("apply() should be implemented by subclasses")


class Identity(Augmentation):
    def apply(self, audio: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        return audio


class TemporalMask(Augmentation):
    def __init__(
        self, prob: float, *, min_length: int, max_length: int, use_noise: bool
    ):
        """Replaces a random interval of all channels of the input with zeros or white noise.
        Does not mutate the input tensor in place.

        Args:
            prob (float): Probability of applying the augmentation to each member of a batch.
            min_length (int): Minimum length of the masked interval in samples.
            max_length (int): Maximum length of the masked interval in samples.
            use_noise (bool): If True, the masked interval is filled with white noise. Otherwise, it is filled with zeros.

        Raises:
            ValueError: If min_length is greater than max_length
        """
        super(TemporalMask, self).__init__(prob)

        self.min_length = min_length
        self.max_length = max_length
        self.use_noise = use_noise

    def apply(self, audio: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """Applies the temporal masking augmentation to the input audio.

        Args:
            audio (torch.Tensor): Audio tensor with shape (batch, samples, channels)
            batch_mask (torch.Tensor): Mask of shape (batch,) indicating which samples to augment.
            A value of 1 (True) indicates the augmentation should be applied to that sample.

        Returns:
            torch.Tensor: The augmented audio signal. Shape does not change

        Raises:
            ValueError: If min_length is greater than the number of samples in the audio tensor
        """
        if self.min_length >= audio.shape[1]:
            raise ValueError(
                f"Audio tensor must be at least as long as min_length. {audio.shape[1]} !>= {self.min_length}"
            )

        audio = audio.clone()

        for i, mask in enumerate(batch_mask):
            if mask:
                start_idx = np.random.randint(0, audio.shape[1] - self.min_length)
                length = np.random.randint(self.min_length, self.max_length + 1)
                end_idx = min(start_idx + length, audio.shape[1])
                if self.use_noise:
                    noise_mag = torch.std(audio[i, start_idx:end_idx, :])
                    audio[i, start_idx:end_idx, :] = (
                        torch.randn_like(audio[i, start_idx:end_idx]) * noise_mag
                    )
                else:
                    audio[i, start_idx:end_idx, :] = 0

        return audio


class ChannelMasking(Augmentation):
    def __init__(
        self, prob: float, *, min_channels: int, max_channels: int, use_noise: bool
    ):
        """Replaces a random subset of channels of the input with zeros or noise.
        Does not mutate the input tensor in place.

        Args:
            prob (float): Probability of applying the augmentation to each member of a batch.
            min_channels (int): Minimum number of channels to mask.
            max_channels (int): Maximum number of channels to mask.
            use_noise (bool): If True, the masked channels are filled with white noise. Otherwise, they are filled with zeros.

        Raises:
            ValueError: If min_channels is greater than max_channels or less than 1
        """
        super(ChannelMasking, self).__init__(prob)

        if min_channels < 1 or min_channels > max_channels:
            raise ValueError(
                f"min_channels must be in the range [1, max_channels], got {min_channels}, {max_channels}"
            )

        self.min_channels = min_channels
        self.max_channels = max_channels
        self.use_noise = use_noise

    def apply(self, audio: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """Applies the channel masking augmentation to the input audio.

        Args:
            audio (torch.Tensor): Audio tensor with shape (batch, samples, channels)
            batch_mask (torch.Tensor): Mask of shape (batch,) indicating which samples to augment.
            A value of 1 (True) indicates the augmentation should be applied to that sample.

        Returns:
            torch.Tensor: The augmented audio signal. Shape does not change

        Raises:
            ValueError: If min_channels is greater than the number of channels in the audio tensor
        """
        if self.min_channels >= audio.shape[2]:
            raise ValueError(
                f"Audio tensor must have at least as many channels as min_channels. {audio.shape[2]} !>= {self.min_channels}"
            )

        # TODO: this can probably be optimized with torch.gather syntax
        new_audio = torch.empty_like(audio)

        for i, mask in enumerate(batch_mask):
            if not mask:
                new_audio[i] = audio[i]
            else:
                n_channels = np.random.randint(self.min_channels, self.max_channels + 1)
                channel_indices = np.random.choice(
                    audio.shape[2], n_channels, replace=False
                )
                complement = np.setdiff1d(np.arange(audio.shape[2]), channel_indices)
                if self.use_noise:
                    noise_mag = torch.std(audio[i])
                    new_audio[i, :, channel_indices] = (
                        torch.randn_like(audio[i, :, channel_indices]) * noise_mag
                    )
                else:
                    audio[i, :, channel_indices] = 0
                new_audio[i, :, complement] = audio[i, :, complement]

        return new_audio


class NoiseInjection(Augmentation):
    def __init__(self, prob: float, *, snr_min: float, snr_max: float):
        """Adds white noise to all channels of the input.

        Args:
            prob (float): Probability of applying the augmentation to each member of a batch.
            snr (float): Signal-to-noise ratio in dB. The higher the value, the lower the noise level.
        """
        super(NoiseInjection, self).__init__(prob)
        self.snr_min = snr_min
        self.snr_max = snr_max

    def apply(self, audio: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """Applies the noise injection augmentation to the input audio.

        Args:
            audio (torch.Tensor): Audio tensor with shape (batch, samples, channels)
            batch_mask (torch.Tensor): Mask of shape (batch,) indicating which samples to augment.
            A value of 1 (True) indicates the augmentation should be applied to that sample.

        Returns:
            torch.Tensor: The augmented audio signal. Shape does not change
        """

        # Sample an SNR for each batch member
        orig_std = torch.std(audio, dim=(1, 2), keepdim=True)
        batch_snr = (
            torch.rand((len(audio),), device=audio.device)
            * (self.snr_max - self.snr_min)
            + self.snr_min
        )[:, None, None]
        noise_mag = orig_std / 10 ** (batch_snr / 20)  # shape: (batch, 1, 1)
        noise_mag = noise_mag * batch_mask[:, None, None]
        audio = audio + torch.randn_like(audio) * noise_mag  # Does not mutate

        return audio / audio.std(dim=(1, 2), keepdim=True) * orig_std


class AugmentationConfig:
    should_use_augmentations: bool
    mask_temporal_prob: float
    mask_temporal_min_length: int
    mask_temporal_max_length: int
    mask_temporal_use_noise: bool
    mask_channels_prob: float
    mask_channels_min_channels: int
    mask_channels_max_channels: int
    mask_channels_use_noise: bool
    noise_injection_prob: float
    noise_injection_snr_min: float
    noise_injection_snr_max: float

    # list args explicitly to allow ide to provide auto-completion
    def __init__(
        self,
        should_use_augmentations: bool = True,
        mask_temporal_prob: float = 0.5,
        mask_temporal_min_length: int = 256,
        mask_temporal_max_length: int = 768,
        mask_temporal_use_noise: bool = True,
        mask_channels_prob: float = 0.5,
        mask_channels_min_channels: int = 1,
        mask_channels_max_channels: int = 2,
        mask_channels_use_noise: bool = True,
        noise_injection_prob: float = 0.5,
        noise_injection_snr_min: float = 3,
        noise_injection_snr_max: float = 12,
    ):
        """Data class which wraps a dictionary of arguments while providing type hints.

        Args:
            should_use_augmentations (bool): Whether to use augmentations.
            mask_temporal_prob (float): Probability of applying temporal masking.
            mask_temporal_min_length (int): Minimum length of the masked interval in samples.
            mask_temporal_max_length (int): Maximum length of the masked interval in samples.
            mask_temporal_use_noise (bool): Whether to fill the masked interval with noise.
            mask_channels_prob (float): Probability of applying channel masking.
            mask_channels_min_channels (int): Minimum number of channels to mask.
            mask_channels_max_channels (int): Maximum number of channels to mask.
            mask_channels_use_noise (bool): Whether to fill the masked channels with noise.
            noise_injection_prob (float): Probability of applying noise injection.
            noise_injection_snr_min (float): Minimum SNR of the injected noise.
            noise_injection_snr_max (float): Maximum SNR of the injected noise
        """
        self.should_use_augmentations = should_use_augmentations
        self.mask_temporal_prob = mask_temporal_prob
        self.mask_temporal_min_length = mask_temporal_min_length
        self.mask_temporal_max_length = mask_temporal_max_length
        self.mask_temporal_use_noise = mask_temporal_use_noise
        self.mask_channels_prob = mask_channels_prob
        self.mask_channels_min_channels = mask_channels_min_channels
        self.mask_channels_max_channels = mask_channels_max_channels
        self.mask_channels_use_noise = mask_channels_use_noise
        self.noise_injection_prob = noise_injection_prob
        self.noise_injection_snr_min = noise_injection_snr_min
        self.noise_injection_snr_max = noise_injection_snr_max


def build_augmentations(
    augmentation_config: AugmentationConfig,
) -> nn.Module:
    """Builds a nn.Module containing the augmentations specified in the config.

    Args:
        augmentation_config (AugmentationConfig): Configuration object specifying the augmentations to use.

    Returns:
        nn.Module: A nn.Module containing the specified augmentations.
    """
    augmentations = []

    if not augmentation_config.should_use_augmentations:
        return Identity(1.0)

    if augmentation_config.noise_injection_prob > 0:
        augmentations.append(
            NoiseInjection(
                augmentation_config.noise_injection_prob,
                snr_min=augmentation_config.noise_injection_snr_min,
                snr_max=augmentation_config.noise_injection_snr_max,
            )
        )

    if augmentation_config.mask_channels_prob > 0:
        augmentations.append(
            ChannelMasking(
                augmentation_config.mask_channels_prob,
                min_channels=augmentation_config.mask_channels_min_channels,
                max_channels=augmentation_config.mask_channels_max_channels,
                use_noise=augmentation_config.mask_channels_use_noise,
            )
        )

    if augmentation_config.mask_temporal_prob > 0:
        augmentations.append(
            TemporalMask(
                augmentation_config.mask_temporal_prob,
                min_length=augmentation_config.mask_temporal_min_length,
                max_length=augmentation_config.mask_temporal_max_length,
                use_noise=augmentation_config.mask_temporal_use_noise,
            )
        )

    return nn.Sequential(*augmentations)


if __name__ == "__main__":
    # demo of augmentations
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_audio(audio: torch.Tensor, title: str):
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        fig.suptitle(title)
        for n in range(4):
            ax = fig.add_subplot(gs[n // 2, n % 2])
            ax.specgram(audio[0, :, n].numpy(), Fs=125000)
            ax.set_title(f"Channel {n}")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    # Sample audio
    duration = 0.080
    sr = 125000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    frequency = 5000 * np.sin(2 * np.pi * t / 0.050) + 25000
    phase = np.cumsum(2 * np.pi * frequency / sr)
    audio = np.sin(phase).reshape(1, -1, 1)
    audio = np.repeat(audio, 4, axis=-1)
    audio = torch.tensor(audio, dtype=torch.float32)

    # Apply augmentations
    for i in range(8):
        use_temp = i & 1
        use_chan = i & 2
        use_noise = i & 4

        augmentations = build_augmentations(
            AugmentationConfig(
                mask_temporal_prob=1.0 if use_temp else 0.0,
                mask_channels_prob=1.0 if use_chan else 0.0,
                noise_injection_prob=1.0 if use_noise else 0.0,
            )
        )

        augmented_audio = augmentations(audio)
        plot_audio(
            augmented_audio,
            f"Temporal: {use_temp > 0}, Channel: {use_chan > 0}, Noise: {use_noise > 0}",
        )
