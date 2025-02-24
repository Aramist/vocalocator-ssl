from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, autograd, nn
from torch.distributions import LowRankMultivariateNormal, MultivariateNormal
from torch.nn import functional as F

from .gvae import GerbilizerVAE


def time_domain_reconstruction_loss(target: Tensor, output: Tensor) -> Tensor:
    """Computes the L1 loss in the time domain without reduction. Assumes the input is always batched."""
    is_multi_channel = len(target.shape) == 3
    if is_multi_channel:
        loss = torch.abs(target - output).mean(dim=(1, 2))
    else:
        loss = torch.abs(target - output).mean(dim=1)
    return loss


def frequency_domain_multi_scale_loss(
    target: Tensor,
    output: Tensor,
    nffts: List[int],
    hop_sizes: List[int],
    l2_weight: float,
) -> Tensor:
    """Computes a loss between two (potentially multichannel) audio signals in the frequency domain.
    The loss is computed as a convex combination between the L1 and L2 losses over the spectrograms.
    """
    if l2_weight < 0 or l2_weight > 1:
        raise ValueError(f"L2 weight should be between 0 and 1, received {l2_weight}")
    if len(nffts) != len(hop_sizes):
        raise ValueError(
            f"nffts and hop_sizes must be the same length, received {nffts} and {hop_sizes}"
        )

    is_multi_channel = len(target.shape) == 3
    if is_multi_channel:
        target = target.transpose(-1, -2)  # Convert to (batch, channels, time)
        output = output.transpose(-1, -2)
        bsz, n_channels, _ = target.shape
        # flatten channel dimension into batch dim
        target = target.reshape(bsz * n_channels, -1)
        output = output.reshape(bsz * n_channels, -1)
    else:
        bsz, _ = target.shape
        n_channels = 1

    l1_loss = torch.zeros((bsz,), device=target.device, dtype=target.dtype)
    l2_loss = torch.zeros((bsz,), device=target.device, dtype=target.dtype)
    for nfft, hop_size in zip(nffts, hop_sizes):
        # Compute the STFTs, shape: (batch * channels, freq, time)
        freq_domain_target = torch.stft(
            target, n_fft=nfft, hop_length=hop_size, return_complex=True
        )
        freq_domain_output = torch.stft(
            output, n_fft=nfft, hop_length=hop_size, return_complex=True
        )

        # Since l2 uses log-power spectrogram, compute it before converting to real/imag
        eps = 1e-8
        l2_term = (
            torch.square(
                torch.log(torch.abs(freq_domain_target) + eps)
                - torch.log(torch.abs(freq_domain_output) + eps)
            )
            .reshape(bsz, n_channels, nfft // 2 + 1, -1)
            .mean(dim=(1, 2, 3))
        )

        # convert from complex to stacked float32 real/imag
        freq_domain_target = torch.stack(
            (torch.real(freq_domain_target), torch.imag(freq_domain_target)), dim=-1
        ).reshape(bsz, n_channels, nfft // 2 + 1, -1, 2)
        freq_domain_output = torch.stack(
            (torch.real(freq_domain_output), torch.imag(freq_domain_output)), dim=-1
        ).reshape(bsz, n_channels, nfft // 2 + 1, -1, 2)
        # new shape: (batch, channels, freq, time, 2)

        # shape: (batch,)
        l1_term = torch.abs(freq_domain_target - freq_domain_output).mean(
            dim=(1, 2, 3, 4)
        )

        l1_loss = l1_loss + l1_term
        l2_loss = l2_loss + l2_term
    l1_loss = l1_loss / len(nffts)
    l2_loss = l2_loss / len(nffts)
    return l1_loss * (1 - l2_weight) + l2_loss * l2_weight


def kl_divergence_loss(mu: Tensor, cov_diag: Tensor, cov_factors: Tensor) -> Tensor:
    prior = MultivariateNormal(
        torch.zeros_like(mu), torch.eye(mu.shape[-1], device=mu.device)
    )
    posterior = LowRankMultivariateNormal(mu, cov_factors, cov_diag)
    return torch.distributions.kl_divergence(posterior, prior)


class MultiScaleDiscriminatorSubnetwork(nn.Module):
    def __init__(self, num_channels: int, nfft: int, hop_size: int):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.activation = F.leaky_relu

        # First int in kernel is for height (freq), second is for width (time)
        # Paper describes an increasing dilation over frequency and a stride of 2 over time
        self.convs = nn.ModuleList(
            [
                # multiply by two to account for real and imag channels
                nn.Conv2d(num_channels * 2, 32, kernel_size=(3, 9), stride=(1, 1)),
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 1), dilation=(1, 1)),
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), dilation=(2, 1)),
                nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), dilation=(4, 1)),
                nn.Conv2d(32, 32, kernel_size=(3, 3)),
                nn.Conv2d(32, 1, kernel_size=(3, 3)),
            ]
        )

    def _make_spectrogram(self, audio: Tensor):
        has_multi_channel = len(audio.shape) == 3
        if has_multi_channel:
            bsz, _, n_channels = audio.shape
            audio = audio.transpose(-1, -2)  # Convert to (batch, channels, time)
        else:
            bsz, _ = audio.shape
            n_channels = 1

        audio = audio.reshape(bsz * n_channels, -1)
        spec = torch.stft(
            audio,
            n_fft=self.nfft,
            hop_length=self.hop_size,
            return_complex=True,
        )
        # shape: (batch * channels, freq, time)
        # Reshape to (batch, channels, freq, time) where frequency acts as the image height
        spec = spec.reshape(bsz, n_channels, self.nfft // 2 + 1, -1)
        # convert from complex to stacked float32 real/imag
        spec = torch.concatenate((spec.real, spec.imag), dim=1)
        # New shape: (batch, channels * 2, freq, time)

        return spec

    def forward(self, audio: Tensor) -> List[Tensor]:
        """Returns a list of discriminator outputs at each scale"""
        spec = self._make_spectrogram(audio)
        outputs = []
        for conv in self.convs:
            spec = self.activation(conv(spec))
            outputs.append(spec)
        outputs.append(
            torch.sigmoid(
                F.adaptive_avg_pool2d(spec, output_size=(1, 1)).reshape(-1, 1, 1, 1)
            )
        )
        return outputs

    def compute_losses(self, orig_audio: Tensor, gen_audio: Tensor):
        """Computes the adversarial and feature matching loss terms for the discriminator
        The adversarial loss term acts as a perceptual loss by motivating the discriminators to
        differentiate generated samples from the original audio
        There is also a separate loss term for training the discriminators alone
        The feature maching loss term pushes the features learned in each layer of each individual
        discriminator to be similar
        """

        outputs_x = self.forward(orig_audio)
        outputs_x, final_output_x = outputs_x[:-1], outputs_x[-1]
        outputs_xhat = self.forward(gen_audio)
        outputs_xhat, final_output_xhat = outputs_xhat[:-1], outputs_xhat[-1]

        feat_loss = None
        # each layer has output of shape (batch, channels, freq (height), time (width))
        for layer_idx in range(len(outputs_x)):
            layer_loss = torch.abs(outputs_x[layer_idx] - outputs_xhat[layer_idx]).sum(
                dim=(1, 2, 3)
            ) / torch.abs(outputs_x[layer_idx]).mean(dim=(1, 2, 3))
            # shape: (batch,)
            if feat_loss is None:
                feat_loss = layer_loss
            else:
                feat_loss = feat_loss + layer_loss
        feat_loss = feat_loss / len(outputs_x)

        disc_loss = F.relu(1 - final_output_x.squeeze()) + F.relu(
            1 + final_output_xhat.squeeze()
        )

        adv_loss = F.relu(1 - final_output_xhat.squeeze())

        return feat_loss, disc_loss, adv_loss


class MultiScaleDiscriminators(nn.Module):
    def __init__(self, num_mics: int, nffts: List[int]):
        super().__init__()
        self.nffts = nffts

        self.discriminators = nn.ModuleList(
            [
                MultiScaleDiscriminatorSubnetwork(
                    num_channels=num_mics, nfft=nfft, hop_size=nfft // 4
                )
                for nfft in nffts
            ]
        )

    def compute_losses(self, orig_audio: Tensor, gen_audio: Tensor):
        """Computes the adversarial and feature matching loss terms for a set of discriminators
        Returns (feat_loss, disc_loss, adv_loss), all of shape (batch,)

        feat_loss: Feature matching loss term for all discriminators
        disc_loss: Discriminator objective function
        adv_loss: Adversarial loss term for generator
        """
        feat_loss = None
        disc_loss = None
        adv_loss = None
        for discriminator in self.discriminators:
            (
                discriminator_feat_loss,
                discriminator_disc_loss,
                discriminator_adv_loss,
            ) = discriminator.compute_losses(orig_audio, gen_audio)
            if feat_loss is None:
                feat_loss = discriminator_feat_loss
                disc_loss = discriminator_disc_loss
                adv_loss = discriminator_adv_loss
            else:
                feat_loss = feat_loss + discriminator_feat_loss
                disc_loss = disc_loss + discriminator_disc_loss
                adv_loss = adv_loss + discriminator_adv_loss
        feat_loss = feat_loss / len(self.discriminators)
        disc_loss = disc_loss / len(self.discriminators)
        adv_loss = adv_loss / len(self.discriminators)
        return feat_loss, disc_loss, adv_loss


class LossHandler(nn.Module):
    def __init__(
        self,
        hyperparams: dict,
        discriminators: MultiScaleDiscriminators,
        balance_losses: bool = False,
    ):
        super().__init__()
        self.hyperparams = hyperparams
        self.discriminators = discriminators
        self.balance_losses = balance_losses

        self.discriminator_update_prob = hyperparams["DISCRIMINATOR_UPDATE_PROB"]
        self.exp_avg_beta = hyperparams["EXP_AVG_BETA"]
        self.freq_loss_nffts = hyperparams["FREQ_LOSS_NFFTS"]

        self.time_domain_weight = hyperparams["TIME_WEIGHT"]
        self.frequency_domain_weight = hyperparams["FREQ_WEIGHT"]
        self.kl_divergence_weight = hyperparams["KL_WEIGHT"]
        self.feature_matching_weight = hyperparams["FEATURE_MATCHING_WEIGHT"]
        self.discriminator_weight = hyperparams["DISCRIMINATOR_WEIGHT"]
        self.adversarial_weight = hyperparams["ADVERSARIAL_WEIGHT"]

        # Reweight all weights to sum to 1
        weight_sum = (
            self.time_domain_weight
            + self.frequency_domain_weight
            + self.kl_divergence_weight
            + self.feature_matching_weight
            + self.discriminator_weight
            + self.adversarial_weight
        )
        self.time_domain_weight /= weight_sum
        self.frequency_domain_weight /= weight_sum
        self.kl_divergence_weight /= weight_sum
        self.feature_matching_weight /= weight_sum
        self.discriminator_weight /= weight_sum
        self.adversarial_weight /= weight_sum

        # Save buffers for running averages of loss gradients
        time_grad_buffer = torch.ones(1)
        freq_grad_buffer = torch.ones(1)
        kl_grad_buffer = torch.ones(1)
        feat_grad_buffer = torch.ones(1)
        adv_grad_buffer = torch.ones(1)
        self.register_buffer("time_grad_buffer", time_grad_buffer)
        self.register_buffer("freq_grad_buffer", freq_grad_buffer)
        self.register_buffer("kl_grad_buffer", kl_grad_buffer)
        self.register_buffer("feat_grad_buffer", feat_grad_buffer)
        self.register_buffer("adv_grad_buffer", adv_grad_buffer)

    def get_loss(
        self, orig_audio: Tensor, model_output: GerbilizerVAE.ReturnType
    ) -> Tuple[Tensor, Tuple]:
        discriminator_losses = self.discriminators.compute_losses(
            orig_audio=orig_audio, gen_audio=model_output.reconstruction
        )
        kl_loss = kl_divergence_loss(
            mu=model_output.mu,
            cov_diag=model_output.cov_diag,
            cov_factors=model_output.cov_factors,
        )
        feat_loss, disc_loss, adv_loss = discriminator_losses
        time_domain_loss = time_domain_reconstruction_loss(
            orig_audio, model_output.reconstruction
        )
        freq_domain_loss = frequency_domain_multi_scale_loss(
            orig_audio,
            model_output.reconstruction,
            nffts=self.freq_loss_nffts,
            hop_sizes=[nfft // 4 for nfft in self.freq_loss_nffts],
            l2_weight=0.5,  # equal weighting between L1 and L2
        )

        return self.balance_and_aggregate_losses(
            reconstruction=model_output.reconstruction,
            time_domain_loss=time_domain_loss.mean(),
            frequency_domain_loss=freq_domain_loss.mean(),
            kl_divergence=kl_loss.mean(),
            feature_matching_loss=feat_loss.mean(),
            discriminator_loss=disc_loss.mean(),
            adversarial_loss=adv_loss.mean(),
        )

    def balance_and_aggregate_losses(
        self,
        reconstruction: Tensor,
        time_domain_loss: Tensor,
        frequency_domain_loss: Tensor,
        kl_divergence: Tensor,
        feature_matching_loss: Tensor,
        discriminator_loss: Tensor,
        adversarial_loss: Tensor,
    ) -> Tuple[Tensor, Tuple]:
        """Balances each loss term according to a running average of it's gradient magnitude.
        This has the effect of balancing all loss terms to be on the same scale.

        Assumes all input tensors are scalar losses
        """
        should_update_discriminator = torch.rand(1) < self.discriminator_update_prob
        should_update_discriminator = should_update_discriminator.to(
            reconstruction.device
        ).to(
            reconstruction.dtype
        )  # this should make it 0 or 1 numerically
        if not self.balance_losses:
            return (
                (
                    self.time_domain_weight * time_domain_loss
                    + self.frequency_domain_weight * frequency_domain_loss
                    + self.kl_divergence_weight * kl_divergence
                    + self.feature_matching_weight * feature_matching_loss
                    + (
                        self.discriminator_weight
                        * discriminator_loss
                        * should_update_discriminator
                    )
                    + self.adversarial_weight * adversarial_loss
                ),
                (
                    time_domain_loss,
                    frequency_domain_loss,
                    kl_divergence,
                    feature_matching_loss,
                    discriminator_loss,
                    adversarial_loss,
                ),
            )

        # First compute the balanced loss with the existing buffer values

        balanced_loss = (
            self.time_domain_weight * time_domain_loss / self.time_grad_buffer
            + self.frequency_domain_weight
            * frequency_domain_loss
            / self.freq_grad_buffer
            + self.kl_divergence_weight * kl_divergence / self.kl_grad_buffer
            + self.feature_matching_weight
            * feature_matching_loss
            / self.feat_grad_buffer
            + self.discriminator_weight
            * discriminator_loss
            * should_update_discriminator
            + self.adversarial_weight * adversarial_loss / self.adv_grad_buffer
        )
        # No buffer for adversarial because it's not a function of generator

        autograd.backward(time_domain_loss, inputs=(reconstruction,), retain_graph=True)
        time_loss_grad = reconstruction.grad
        time_loss_grad = torch.linalg.norm(time_loss_grad)
        reconstruction.grad = None

        autograd.backward(
            frequency_domain_loss, inputs=(reconstruction,), retain_graph=True
        )
        freq_loss_grad = reconstruction.grad
        freq_loss_grad = torch.linalg.norm(freq_loss_grad)
        reconstruction.grad = None

        # autograd.backward(kl_divergence, inputs=(reconstruction,))
        # kl_loss_grad = reconstruction.grad
        # kl_loss_grad = torch.linalg.norm(kl_loss_grad)
        # reconstruction.grad = None

        autograd.backward(
            feature_matching_loss, inputs=(reconstruction,), retain_graph=True
        )
        feat_loss_grad = reconstruction.grad
        feat_loss_grad = torch.linalg.norm(feat_loss_grad)
        reconstruction.grad = None

        autograd.backward(adversarial_loss, inputs=(reconstruction,), retain_graph=True)
        adv_loss_grad = reconstruction.grad
        adv_loss_grad = torch.linalg.norm(adv_loss_grad)
        reconstruction.grad = None

        # Update the buffers
        self.time_grad_buffer = (
            self.exp_avg_beta * self.time_grad_buffer
            + (1 - self.exp_avg_beta) * time_loss_grad
        )
        self.freq_grad_buffer = (
            self.exp_avg_beta * self.freq_grad_buffer
            + (1 - self.exp_avg_beta) * freq_loss_grad
        )
        self.feat_grad_buffer = (
            self.exp_avg_beta * self.feat_grad_buffer
            + (1 - self.exp_avg_beta) * feat_loss_grad
        )
        self.adv_grad_buffer = (
            self.exp_avg_beta * self.adv_grad_buffer
            + (1 - self.exp_avg_beta) * adv_loss_grad
        )

        print(
            f"buffers: {self.time_grad_buffer}, {self.freq_grad_buffer}, {self.feat_grad_buffer}, {self.adv_grad_buffer}"
        )

        return balanced_loss
