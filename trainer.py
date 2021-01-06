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

""" Contains high-level modules for training adversarial text-to-speech models. """

import gc
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import mu_inverse, mu_transform, print_batch_stats, print_list_values


class Trainer(nn.Module):
    """ Trainer for generative adversarial text-to-speech model. """

    def __init__(
        self,
        sampling_rate: int,
        audio_generator: DictConfig,
        audio_discriminator: DictConfig,
        spect_discriminator: DictConfig,
        spect_fn: DictConfig,
        sdtw_fn: DictConfig,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.audio_generator = instantiate(audio_generator)
        self.audio_discriminator = instantiate(audio_discriminator)
        self.spect_discriminator = instantiate(spect_discriminator)
        self.spect_fn = instantiate(spect_fn)
        self.sdtw_fn = instantiate(sdtw_fn)

    def d_loss(self, y_d, y_pred_d, z_d, z_pred_d, debug: bool = False):
        """ Computes loss for discriminators. """
        l_real = sum(((pred - 1.) ** 2).mean(0).sum() for pred in y_d + z_d)
        l_fake = sum((pred ** 2).mean(0).sum() for pred in y_pred_d + z_pred_d)

        if debug:
            print_list_values(l_real, l_fake)

        return {"real": l_real, "fake": l_fake}

    def g_loss(self, x_latents, y_len, y_pred_g, y_pred_len, z, z_pred, z_pred_g, debug: bool = False):
        """ Computes loss for generator. """
        l_nll = 0.5 * (x_latents ** 2).mean()
        l_adv = sum(((pred - 1.) ** 2).mean(0).sum() for pred in y_pred_g + z_pred_g)
        l_hard = F.l1_loss(z, z_pred)
        l_soft = self.sdtw_fn(z, z_pred)
        l_mse = 0.5 * ((y_len.float() - y_pred_len.float()) ** 2).mean()

        if debug:
            print_list_values(l_nll, l_adv, l_hard, l_soft, l_mse)

        return {"nll": l_nll, "adv": l_adv, "hard": l_hard, "soft": l_soft, "mse": l_mse}

    def d_step(self, x, x_len, y, y_len, y_offset, aligner_len, jitter_steps: int = 0, debug: bool = False):
        """ Computes one step through the discriminator. """
        with torch.no_grad():
            x_latents, y_pred, y_pred_len = self.audio_generator(x, x_len, y_len=aligner_len, y_offset=y_offset)
            z_pred = self.spect_fn(mu_inverse(y_pred), jitter_steps=0)
            z = self.spect_fn(y, jitter_steps=jitter_steps)
            y = mu_transform(y)

        if debug:
            print_batch_stats(y, prefix="y*\t")
            print_batch_stats(y_pred, prefix="y^\t")
            print_batch_stats(x_latents, prefix="xl\t")

        y_d = self.audio_discriminator(y)
        y_pred_d = self.audio_discriminator(y_pred)
        z_d = self.spect_discriminator(z)
        z_pred_d = self.spect_discriminator(z_pred)

        loss_dict = self.d_loss(y_d, y_pred_d, z_d, z_pred_d, debug=debug)
        gc.collect()
        return loss_dict

    def g_step(self, x, x_len, y, y_len, y_offset, aligner_len, jitter_steps: int = 0, debug: bool = False):
        """ Computes one step through the generator. """
        x_latents, y_pred, y_pred_len = self.audio_generator(x, x_len, y_len=aligner_len, y_offset=y_offset)
        z_pred = self.spect_fn(mu_inverse(y_pred), jitter_steps=0)

        with torch.no_grad():
            z = self.spect_fn(mu_transform(y), jitter_steps=jitter_steps)

        y_pred_g = self.audio_discriminator(y_pred)
        z_pred_g = self.spect_discriminator(z_pred)

        loss_dict = self.g_loss(x_latents, y_len, y_pred_g, y_pred_len, z, z_pred, z_pred_g, debug=debug)
        gc.collect()
        return loss_dict

    def step(self, x, x_len, y, y_len, y_offset, aligner_len, jitter_steps: int = 60, debug: bool = False):
        """ Completes one full step through the model. """
        x_latents, y_pred, y_pred_len = self.audio_generator(x, x_len, y_len=aligner_len, y_offset=y_offset)
        z_pred = self.spect_fn(mu_inverse(y_pred), jitter_steps=0)

        with torch.no_grad():
            z = self.spect_fn(y, jitter_steps=jitter_steps)
            y = mu_transform(y)

        if debug:
            print_batch_stats(y, prefix="y*\t")
            print_batch_stats(y_pred, prefix="y^\t")
            print_batch_stats(x_latents, prefix="xl\t")

        y_d = self.audio_discriminator(y)
        y_pred_d = self.audio_discriminator(y_pred)
        z_d = self.spect_discriminator(z)
        z_pred_d = self.spect_discriminator(z_pred)

        y_pred_g = self.audio_discriminator(y_pred)
        z_pred_g = self.audio_discriminator(z_pred)

        d_loss_dict = self.d_loss(y_d, y_pred_d, z_d, z_pred_d, debug=debug)
        g_loss_dict = self.g_loss(x_latents, y_len, y_pred_g, y_pred_len, z, z_pred, z_pred_g, debug=debug)

        return d_loss_dict, g_loss_dict, y_pred, z_pred, z

    def print_model_summary(self):
        """ Outputs summary of module parameters. """
        labels = ["", "K", "M", "B", "T"]
        headers = ["", "Name", "Type", "Params"]
        table = []
        i = 0

        def get_readable_number(number):
            num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
            num_groups = int(np.ceil(num_digits / 3))
            num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
            shift = -3 * (num_groups - 1)
            number = number * (10 ** shift)
            index = num_groups - 1
            if index < 1 or number >= 100:
                return "{:d} {}".format(int(number), labels[index])
            else:
                return "{:.1f} {}".format(number, labels[index])

        def get_parameter_count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        for k, v in self._modules.items():
            if not isinstance(v, nn.Module):
                continue
            table += [[i, k, v.__class__.__name__, get_readable_number(get_parameter_count(v))]]
        print(tabulate(table, headers=headers))

    @staticmethod
    @torch.no_grad()
    def apply_orthogonal_regularization(parameters, weight: float = 1e-4):
        """ Computes and applies off-diagonal orthogonal regularization directly to specified parameters."""
        for param in parameters:
            if len(param.shape) < 2 or param.grad is None:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += weight * grad.view(param.shape)

    @property
    def discriminator_parameters(self):
        return list(self.audio_discriminator.parameters()) + list(self.spect_discriminator.parameters())

    @property
    def generator_parameters(self):
        return list(self.audio_generator.parameters())

    @property
    def parameters(self):
        return self.discriminator_parameters + self.generator_parameters

