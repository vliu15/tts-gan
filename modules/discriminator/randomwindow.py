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

""" Contains random-window discriminator module for GAN-TTS models. """

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import ResBlock1d, SpectralNormConv1d


class RandomWindowDiscriminator(nn.Module):
    """ Random window discriminator. Reshapes inputs to temporal dim and applies residual layers.

    Args:
        n_layers: number of residual block / downsampling layers
        in_channels: number of channels of input
        base_channels: number of channels in first convolutional layer
        kernel_size: temporal size of convolutional filters
    """

    def __init__(self, n_layers: int, in_channels: int, base_channels: int, kernel_size: int = 3):
        super().__init__()

        self.in_channels = in_channels
        self.n_layers = n_layers

        self.activation = nn.LeakyReLU(0.2)

        self.proj_in = SpectralNormConv1d(in_channels, base_channels, 1)

        channels = base_channels
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks += [
                ResBlock1d(channels, channels, kernel_size=kernel_size, dilation=1, scale_factor=1, activation=self.activation, normalization=None, spectral_norm=True),
                ResBlock1d(channels, channels * 2, kernel_size=kernel_size, dilation=1, scale_factor=0.5, activation=self.activation, normalization=None, spectral_norm=True),
            ]
            channels *= 2

        self.proj_out = SpectralNormConv1d(channels, 1, 1)

    def forward(self, x):
        """
        x: [b, t]
        """
        x = x.unfold(1, self.in_channels, self.in_channels).permute(0, 2, 1)
        x = self.proj_in(x)

        mask = None
        for i in range(self.n_layers):
            x, mask = self.blocks[2 * i](x, mask=mask)
            x, mask = self.blocks[2 * i + 1](x, mask=mask)

        x = self.proj_out(x).squeeze(1)
        return x
