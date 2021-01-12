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

""" Contains decoder module for GAN-TTS models. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from modules.layers import BatchNorm1d, ResBlock1d, SpectralNormConv1d
from modules.utils import sequence_mask


class Decoder(nn.Module):
    """ Waveform decoder.

    Maps sequence of latent vectors into raw audio waveforms. Architecture adapted from generator in GAN-TTS.

    Args:
        n_layers: number of upsampling layers
        in_channels: dimension of latent vectors from the encoder
        hidden_channels: number of channels in first convolutional layer
        kernel_size: temporal size of convolutional filters

    Reference:
    > (Binkowski et al. 2020) High Fidelity Speech Synthesis with Adversarial Networks, https://arxiv.org/abs/1909.11646
    """

    def __init__(
        self,
        scale_factors: List[int] = [2, 3, 3, 5],
        in_channels: int = 256,
        hidden_channels: int = 256,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.scale_factors = scale_factors
        self.n_layers = len(scale_factors)
        self.activation = F.gelu

        self.proj_in = nn.Conv1d(in_channels, hidden_channels, 1)

        channels = hidden_channels
        self.blocks = nn.ModuleList()
        for i, scale_factor in enumerate(scale_factors):
            self.blocks += [
                ResBlock1d(channels, channels, kernel_size=kernel_size, dilation=1, scale_factor=scale_factor, activation=self.activation, normalization=BatchNorm1d, spectral_norm=False),
                ResBlock1d(channels, channels, kernel_size=kernel_size, dilation=4, scale_factor=1, activation=self.activation, normalization=BatchNorm1d, spectral_norm=False),
                ResBlock1d(channels, channels // 2, kernel_size=kernel_size, dilation=16, scale_factor=1, activation=self.activation, normalization=BatchNorm1d, spectral_norm=False),
            ]
            channels //= 2

        self.norm_out = BatchNorm1d(channels)
        self.proj_out = nn.Conv1d(channels, 1, 1)

    def forward(self, y, y_len):
        """
        y: [b, t_y]
        y_len: [b]
        """
        # Create mask for collated latents.
        mask = sequence_mask(y_len, y.size(2)).unsqueeze(1).to(y.dtype)

        # Project to hidden channels.
        y = self.proj_in(y * mask)

        # Apply decoder blocks.
        for i in range(self.n_layers):
            y, mask = self.blocks[3 * i](y, mask=mask)
            y, mask = self.blocks[3 * i + 1](y, mask=mask)
            y, mask = self.blocks[3 * i + 2](y, mask=mask)

        # Project to audio waveform.
        y = self.proj_out(y * mask)
        y = torch.tanh(y * mask)

        return y.squeeze(1), mask
