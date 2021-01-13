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

""" Contains encoder module for GAN-TTS models. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import LayerNorm1d, ResBlock1d
from modules.utils import ones_mask, sequence_mask


class DurationPredictor(nn.Module):
    """ Token duration predictor for the encoder.

    Takes in embeddings of the input tokens and predicts how many frames of audio are aligned
    to each text token. Predicted frame lengths are normalized to 1 for training stability.
    Architecture adapted from the duration predictor in FastSpeech.

    Args:
        hidden_channels: number of channels in input and through the network

    References:
    > (Ren et al. 2019) FastSpeech: Fast, Robust, and Controllable Text to Speech, https://arxiv.org/abs/1905.09263
    > (Kim et al. 2020) Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search, https://arxiv.org/abs/2005.11129
    """

    def __init__(self, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)

        self.norm1 = LayerNorm1d(hidden_channels)
        self.norm2 = LayerNorm1d(hidden_channels)
        self.norm3 = LayerNorm1d(hidden_channels)

        self.activation = F.gelu

        self.proj_out = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, mask=None):
        if mask is None:
            mask = ones_mask(x)

        x = self.norm1(x, mask=mask)
        x = self.activation(x)
        x = self.conv1(x * mask)

        x = self.norm2(x, mask=mask)
        x = self.activation(x)
        x = self.conv2(x * mask)

        x = self.norm3(x, mask=mask)
        x = self.activation(x)
        x = self.conv3(x * mask)

        x = self.proj_out(x * mask)
        x = x.exp() * mask

        return x.squeeze(1)


class Encoder(nn.Module):
    """ Phonome encoder.

    Produces latent representations to be fed to aligner. Same architecture as text encoder in Glow-TTS.

    Args:
        n_vocab: number of unique phoneme inputs to embed
        n_layers: number of dilated residual block layers
        emb_channels: number of channels for each embedded phoneme
        hidden_channels: number of channels throughout the encoder
        out_channels: dimension of latent vectors to pass to the decoder
        kernel_size: temporal size of convolutional filters

    References:
    > (Donahue et al. 2020) End-to-End Adversarial Text-to-Speech, https://arxiv.org/abs/2006.03575
    > (Kim et al. 2020) Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search, https://arxiv.org/abs/2005.11129
    """

    def __init__(
        self,
        n_vocab: int,
        n_layers: int = 10,
        hidden_channels: int = 256,
        out_channels: int = 256,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        self.activation = F.gelu

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks += [
                ResBlock1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=1, scale_factor=1, activation=self.activation, normalization=LayerNorm1d),
                ResBlock1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=4, scale_factor=1, activation=self.activation, normalization=LayerNorm1d),
                ResBlock1d(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=16, scale_factor=1, activation=self.activation, normalization=LayerNorm1d),
            ]

        self.proj_l = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_d = DurationPredictor(hidden_channels, kernel_size=kernel_size)

    def forward(self, x, x_len):
        """
        x: [b, t_x]
        x_len: [b]
        """
        # Embed input text.
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t_x, c]
        x = x.permute(0, 2, 1)  # [b, c, t_x]

        # Create mask for collated text.
        mask = sequence_mask(x_len, x.size(-1)).unsqueeze(1).to(x.dtype)  # [b, 1, t_x]

        # Apply encoder blocks.
        for i in range(self.n_layers):
            x, mask = self.blocks[3 * i](x, mask=mask)
            x, mask = self.blocks[3 * i + 1](x, mask=mask)
            x, mask = self.blocks[3 * i + 2](x, mask=mask)

        # Project to latent and duration variables.
        x_l = self.proj_l(x * mask)
        x_d = self.proj_d(x, mask=mask)

        return x_l * mask, x_d, mask
