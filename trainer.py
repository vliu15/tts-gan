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

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mixture import mean_from_mix_gaussian, mix_gaussian_loss
from modules.utils import get_postprocessing_fn, print_batch_stats, print_cuda_memory, print_list_values


def frobenius_norm(x):
    return torch.sqrt((x.flatten(1) ** 2).sum(1))


def spectral_convergence(i, j):
    return (frobenius_norm(i - j) / frobenius_norm(i)).mean(0)


def log_stft_magnitude(i, j):
    return torch.abs(torch.log(i) - torch.log(j)).sum(-1).mean()


class Trainer(nn.Module):
    """ Trainer for generative adversarial text-to-speech model.

    Args:
        sampling_rate: sampling rate of the audio trained to generate
        audio_generator: structured config for audio generator
        audio_discriminator: structured config for audio discriminator
        spect_fn: structured config for mel-spectrogram computation
        stft_fn: nested structured config for short-time fourier transform function
        mu_law: whether to learn mu-transformed audio directly
    """

    def __init__(
        self,
        sampling_rate: int,
        audio_generator: DictConfig,
        audio_discriminator: DictConfig,
        spect_fn: DictConfig,
        stft_fn: DictConfig,
        mu_law: bool = True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.audio_generator = instantiate(audio_generator)
        self.audio_discriminator = instantiate(audio_discriminator)
        self.spect_fn = instantiate(spect_fn)
        self.stft_fn = nn.ModuleList([instantiate(stft) for stft in stft_fn])
        self.post_fn = get_postprocessing_fn(mu_law=mu_law)

    def d_loss(self, y_d, y_pred_d, debug: bool = False):
        """ Computes loss for discriminators. """
        l_real = sum(((pred - 1) ** 2).sum(-1).mean() for pred in y_d)
        l_fake = sum((pred ** 2).sum(-1).mean() for pred in y_pred_d)

        if debug:
            print_list_values(l_real, l_fake, prefix="d\t")

        return {"real": l_real, "fake": l_fake}

    def g_loss(self, y, y_len, y_pred, y_pred_g, y_pred_len, s, s_pred, debug: bool = False):
        """ Computes loss for generator. """
        l_adv = sum(((pred - 1) ** 2).sum(-1).mean() for pred in y_pred_g)
        l_mse = 0.5 * ((y_len.float() - y_pred_len.float()) ** 2).mean()
        l_sc = sum(spectral_convergence(i, j) for i, j in zip(s, s_pred)) / len(self.stft_fn)
        l_mag = sum(log_stft_magnitude(i, j) for i, j in zip(s, s_pred)) / len(self.stft_fn)
        l_mix = mix_gaussian_loss(y_pred, y)

        if debug:
            print_list_values(l_adv, l_mse, l_sc, l_mag, l_mix, prefix="g\t")

        return {"adv": l_adv, "mse": l_mse, "sc": l_sc, "mag": l_mag, "mix": l_mix}

    def d_step(self, x, x_len, y, y_len, y_offset, aligner_len, debug: bool = False):
        """ Computes one step through the discriminator. """
        with torch.no_grad():
            y_pred, y_pred_len, x_latents, y_latents = self.audio_generator(x, x_len, y_len=aligner_len, y_offset=y_offset)
            y_pred = mean_from_mix_gaussian(y_pred)
            z_pred = self.spect_fn(self.post_fn(y_pred), jitter_steps=0)
            z = self.spect_fn(self.post_fn(y), jitter_steps=0)

            if debug:
                print_cuda_memory()
                print_batch_stats(y, prefix="y*\t")
                print_batch_stats(y_pred, prefix="y^\t")
                print_batch_stats(x_latents, prefix="xl\t")
                print_batch_stats(y_latents, prefix="yl\t")

        y_d = self.audio_discriminator(y)
        y_pred_d = self.audio_discriminator(y_pred)

        loss_dict = self.d_loss(y_d, y_pred_d, debug=debug)

        with torch.no_grad():
            y = self.post_fn(y)
            y_pred = self.post_fn(y_pred)

        return loss_dict, y, y_pred, z, z_pred

    def g_step(self, x, x_len, y, y_len, y_offset, aligner_len, jitter_steps: int = 0, debug: bool = False):
        """ Computes one step through the generator. """
        y_pred, y_pred_len, _, _ = self.audio_generator(x, x_len, y_len=aligner_len, y_offset=y_offset)
        y_samp = mean_from_mix_gaussian(y_pred)
        s_pred = [stft_fn(self.post_fn(y_samp)) for stft_fn in self.stft_fn]

        with torch.no_grad():
            s = [stft_fn(self.post_fn(y)) for stft_fn in self.stft_fn]

        y_pred_g = self.audio_discriminator(y_samp)

        loss_dict = self.g_loss(y, y_len, y_pred, y_pred_g, y_pred_len, s, s_pred, debug=debug)
        return loss_dict

    @torch.no_grad()
    def infer(self, x, x_len=None):
        """ Runs inference on input phoneme sequences. """
        if x_len is None:
            x_len = x.size(-1) * torch.ones(x.size(0), device=x.device).long()
        _, y_pred, _ = self.audio_generator(x, x_len, y_len=None, y_offset=None)
        y_pred = self.post_fn(y_pred)
        return y_pred

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
            i += 1

        print(tabulate(table, headers=headers))

    @staticmethod
    @torch.no_grad()
    def apply_orthogonal_regularization(named_parameters, weight: float = 1e-4):
        """ Computes and applies off-diagonal orthogonal regularization directly to specified parameters."""
        for name, param in named_parameters:
            if len(param.shape) < 2 or param.grad is None or "emb" in name:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += weight * grad.view(param.shape)

    @property
    def discriminator_parameters(self):
        return list(self.audio_discriminator.parameters())

    @property
    def named_discriminator_parameters(self):
        return list(self.audio_discriminator.named_parameters())

    @property
    def generator_parameters(self):
        return list(self.audio_generator.parameters())

    @property
    def named_generator_parameters(self):
        return list(self.audio_generator.named_parameters())

    @property
    def parameters(self):
        return self.discriminator_parameters + self.generator_parameters

