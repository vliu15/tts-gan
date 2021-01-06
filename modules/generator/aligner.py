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

""" Contains aligner module for GAN-TTS models. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    """ Aligner module, which interpolates input latent variables to output latent variables.

    Args:
        gamma: the variance (temperature) of the Gaussian kernel applied to logits before softmax.

    References:
    > (Donahue et al. 2020) End-to-End Adversarial Text-to-Speech, https://arxiv.org/abs/2006.03575
    """

    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x_latents, x_lengths, x_mask, y_len, y_offset=None):
        """
        x_latents: [b, c, t_x]
        x_lengths: [b, t_x]
        x_mask: [b, 1, t_x]
        y_len: [b]
        y_offset: [b]
        """
        if y_offset is None:
            y_offset = torch.zeros_like(y_len)

        x_ends = torch.cumsum(x_lengths, dim=-1)  # [b, t_x]
        x_centers = x_ends - 0.5 * x_lengths  # [b, t_x]

        pos = torch.arange(y_len.max(), device=y_len.device, dtype=y_len.dtype).unsqueeze(0) + y_offset.unsqueeze(1)  # [b, t_y]
        dist = x_centers.unsqueeze(-1) - pos.unsqueeze(1).float()  # [b, t_x, t_y]
        logits = -(dist ** 2 / self.gamma) - 1e9 * (1. - x_mask.permute(0, 2, 1))  # [b, t_x, t_y]

        alignment = F.softmax(logits, dim=1)  # [b, t_x, t_y]
        y_latents = torch.bmm(x_latents, alignment)  # [b, c, t_y]
        return y_latents
