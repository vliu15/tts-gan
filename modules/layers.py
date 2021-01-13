# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" Contains common layers and networks. """

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import ones_mask


class SpectralNormConv1d(nn.Module):
    """ Spectrally-normalized 1d convolution (wrapper). """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv1d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class BatchNorm1d(nn.Module):
    """ Batch normalization for 1d inputs. Supports input masks.

    Args:
        num_features: number of features in input channel
        eps: a value added to the denominator for numerical stability
        momentum: value used for the running_mean and running_var computation. Can be set to None for cumulative moving average
        affine: whether to learn affine parameters
        track_running_stats: whether to tracks the running mean and variance

    References:
    > (Ioffe et al. 2015) Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, https://arxiv.org/abs/1502.03167
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask=None):
        # Calculate the masked mean and variance.
        B, C, L = input.shape
        if mask is not None and mask.shape != (B, 1, L):
            raise ValueError("Mask should have shape (B, 1, L).")
        if C != self.num_features:
            raise ValueError("Expected %d channels but input has %d channels" % (self.num_features, C))
        if mask is not None:
            masked = input * mask
            n = mask.sum()
        else:
            masked = input
            n = B * L
        # Sum.
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask.
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats.
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input.
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters.
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed


class LayerNorm1d(nn.Module):
    """ Layer normalization for 1d inputs. Supports input masks.

    Args:
        channels: number of channels of input
        eps: small float to avoid division by 0

    References:
    > (Ba et al. 2016) Layer Normalization, https://arxiv.org/abs/1607.06450
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, mask=None):
        lens = torch.sum(mask, -1, keepdim=True)
        mean = torch.sum(x * mask, -1, keepdim=True) / lens
        variance = torch.sum(((x - mean) ** 2) * mask, -1, keepdim=True) / lens

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x * self.gamma + self.beta
        return x * mask


class ResBlock1d(nn.Module):
    """ Residual block with option for upsampling or downsampling.
    
    Increases dilation by a factor of 2. Architecture adapted from GBlock and DBlock in GAN-TTS.

    Args:
        in_channels: number of channels of input
        hidden_channels: number of projected / output channels
        kernel_size: temporal size of convolutional filters
        dilation: dilation factor of first convolution (doubled for second)
        scale_factor: factor by which to upsample/downsample the temporal size of input
        activation: activation function to be used between convolutional layers
        spectral_norm: whether to apply spectral norm to convolutions

    References:
    > (Binkowski et al. 2020) High Fidelity Speech Synthesis with Adversarial Networks, https://arxiv.org/abs/1909.11646
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale_factor: int = 1,
        activation: callable = F.relu,
        normalization: nn.Module = None,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.activation = activation

        convolution = SpectralNormConv1d if spectral_norm else nn.Conv1d

        # Select upsampling function based on scale_factor.
        self.upsample = None
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        elif scale_factor < 1:
            self.upsample = nn.AvgPool1d(int(1./scale_factor), int(1./scale_factor))

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = convolution(in_channels, hidden_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = convolution(hidden_channels, hidden_channels, kernel_size, padding=2 * padding, dilation=2 * dilation)

        self.norm1 = None
        self.norm2 = None
        if normalization is not None:
            self.norm1 = normalization(in_channels)
            self.norm2 = normalization(hidden_channels)

        self.proj = convolution(in_channels, hidden_channels, 1)

    def forward(self, x, mask=None):
        """
        x: [b, c, t]
        mask: [b, 1, t]
        """
        if mask is None:
            mask = ones_mask(x)

        # No normalization in discriminator.
        if self.norm1 is not None:
            y = self.norm1(x, mask=mask)
            y = self.activation(y)
        else:
            y = self.activation(x)

        # Upsample before first convolution.
        if self.upsample is not None:
            x = self.upsample(x)
            y = self.upsample(y)
            mask = self.upsample(mask)

        # Apply first set of convolutions.
        y = self.conv1(y * mask)

        # No normalization in discriminator.
        if self.norm2 is not None:
            y = self.norm2(y, mask=mask)
        y = self.activation(y)
        y = self.conv2(y * mask)

        # Project residual.
        x = self.proj(x * mask)

        # Add residual.
        x = x + y

        return x * mask, mask
