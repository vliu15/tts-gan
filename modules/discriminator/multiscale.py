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

""" Contains multi-scale discriminator module for GAN-TTS models. """

import torch.nn as nn

from modules.layers import BatchNorm1d, ResBlock1d, SpectralNormConv1d


class Discriminator(nn.Module):
    """ Single resolution discriminator for 1d tensors. Applies spectral normalization to all convolutions.
    
    Args:
        n_layers: number of residual block layers
        in_channels: number of channels of input
        base_channels: number of channels in first convolutional layer
        kernel_size: temporal size of convolutional filters
    """

    def __init__(self, n_layers: int, in_channels: int, base_channels: int, kernel_size: int):
        super().__init__()
        self.n_layers = n_layers
        channels = base_channels

        self.activation = nn.LeakyReLU(0.2)

        self.proj_in = SpectralNormConv1d(in_channels, channels, 1)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers += [
                ResBlock1d(channels, channels, kernel_size=kernel_size, dilation=1, scale_factor=1, activation=self.activation, normalization=None, spectral_norm=True),
                ResBlock1d(channels, channels * 2, kernel_size=kernel_size, dilation=1, scale_factor=1, activation=self.activation, normalization=None, spectral_norm=True),
            ]
            channels *= 2

        self.proj_out = SpectralNormConv1d(channels, 1, 1)

    def forward(self, x):
        """
        x: [b, c, t]
        """
        x = self.proj_in(x)
        for i in range(self.n_layers):
            x, _ = self.layers[2 * i](x, mask=None)
            x, _ = self.layers[2 * i + 1](x, mask=None)
        x = self.proj_out(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    """ Multi-scale discriminator for 1d tensors.
    
    Args:
        n_discs: number of discriminators (at successfully halved resolutions)
        n_layers: number of residual block layers
        in_channels: number of channels of input
        base_channels: number of channels in first convolutional layer
        kernel_size: temporal size of convolutional filters
    """

    def __init__(self, n_discs: int, n_layers: int, in_channels: int, base_channels: int, kernel_size: int = 3):
        super().__init__()
        channels = base_channels

        self.n_discs = n_discs
        self.downsample = nn.AvgPool1d(2, 2)
        self.discriminators = nn.ModuleList([
            Discriminator(n_layers, in_channels, base_channels, kernel_size) for i in range(n_discs)
        ])

    def forward(self, x):
        """
        x: [b, c, t]
        """
        outs = []
        for i in range(self.n_discs):
            outs += [self.discriminators[i](x)]
            x = self.downsample(x)
        return outs
