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

""" High-level modules for generator and discriminators. """

import math
import random
import torch
import torch.nn as nn
from typing import List

from modules.generator import Aligner, Encoder, Decoder
from modules.discriminator import MultiScaleDiscriminator, RandomWindowDiscriminator
from modules.phonemizer import n_vocab


class AudioGenerator(nn.Module):
    """ Generator module for audio, encodes text, aligns to low-frequency audio, and decodes by upsampling. """

    def __init__(
        self,
        align_gamma: float = 10.0,
        encoder_layers: int = 10,
        encoder_channels: int = 256,
        decoder_scales: List[int] = [2, 3, 3, 5],
        decoder_channels: int = 256,
        latent_channels: int = 256,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.aligner = Aligner(gamma=align_gamma)
        self.encoder = Encoder(
            n_vocab, n_layers=encoder_layers, hidden_channels=encoder_channels, out_channels=latent_channels, kernel_size=kernel_size,
        )
        self.decoder = Decoder(
            scale_factors=decoder_scales, in_channels=latent_channels, hidden_channels=decoder_channels, kernel_size=kernel_size,
        )

    def forward(self, x, x_len, y_len=None, y_offset=None):
        mu, logv, x_lengths, x_mask = self.encoder(x, x_len)
        x_latents = self.sample(mu, logv)
        y_pred_len = x_lengths.sum(-1)

        # Patch in y_len for inference.
        if y_len is None:
            y_len = torch.ceil(y_pred_len)

        y_latents = self.aligner(x_latents, x_lengths, x_mask, y_len, y_offset=y_offset)
        y, y_mask = self.decoder(y_latents, y_len)

        return y, y_pred_len, mu, logv, y_latents

    @staticmethod
    def sample(mu, logv):
        """ Samples from N(mu, v) with the reparameterization trick. """
        eps = torch.randn(mu.size(), dtype=mu.dtype, device=mu.device)
        return mu + torch.exp(logv / 2.) * eps


class AudioDiscriminator(nn.Module):
    """ Discriminator module for audio, applies random window discriminators to evaluate audio. """

    def __init__(
        self,
        windows: List[int] = [200, 400, 800, 1600, 3200],
        n_layers: int = 4,
        base_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.windows = windows
        self.discriminators = nn.ModuleList([
            RandomWindowDiscriminator(n_layers, in_channels, base_channels, kernel_size=kernel_size)
            for in_channels in [window // windows[0] for window in windows]
        ])

    def forward(self, y):
        output = []
        for i, window in enumerate(self.windows):
            start = random.randint(0, y.size(-1) - window)
            output += [self.discriminators[i](y[:, start:start + window])]
        return output


class SpectrogramDiscriminator(nn.Module):
    """ Discriminator module for spectrogram, applies multi-scale discriminators to evaluate spectrograms. """

    def __init__(
        self,
        n_discs: int = 3,
        n_layers: int = 4,
        mel_channels: int = 80,
        base_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.discriminator = MultiScaleDiscriminator(n_discs, n_layers, mel_channels, base_channels, kernel_size=kernel_size)

    def forward(self, z):
        return self.discriminator(z)
