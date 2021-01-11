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

""" Contains auxiliary helper functions. """

import functools
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn


def compose(*fns):
    """ Composes a list of functions. """
    def compose2(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose2, fns, lambda x: x)


def get_preprocessing_fn(mu_law: bool = True):
    """ Returns sequential list of functions for preprocessing audio. """
    fns = []
    if mu_law:
        fns += [mu_transform]
    
    # If no transforms. return identity function.
    if len(fns) == 0:
        return lambda x: x

    return compose(*fns)

def get_postprocessing_fn(mu_law: bool = True):
    """ Returns sequential list of functions for postprocessing audio. """
    fns = []

    if mu_law:
        fns += [mu_inverse]

    # If no transforms. return identity function.
    if len(fns) == 0:
        return lambda x: x

    return compose(*fns)


def seed_everything(seed: int = 1234):
    """ Sets seed for pseudo-random number generators in pytorch, numpy, python.random, os.environ["PYTHONHASHSEED"]. """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        print("{} is not in bounds, numpy accepts from {} to {}".format(seed, min_seed_value, max_seed_value))
        seed = random.randint(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def plot_spectrogram_to_numpy(spectrogram):
    """ Converts spectrogram image as numpy array. """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def sequence_mask(length, max_length=None):
    """ Get masks for given lengths. """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def convert_pad_shape(pad_shape):
    """ Used to get arguments for F.pad. """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


@torch.jit.script
def ones_mask(x):
    """ Creates a mask of ones in the same batch and length size as input. """
    assert len(x.size()) == 3
    return torch.ones((x.size(0), 1, x.size(-1)), dtype=x.dtype, device=x.device)


@torch.jit.script
def mu_transform(x):
    """ Forward operation of mu-law transform for 16-bit integers. """
    assert x.min() >= -1 and x.max() <= 1
    return torch.sign(x) * torch.log(1 + 32768. * torch.abs(x)) / math.log(1 + 32768.)


@torch.jit.script
def mu_inverse(y):
    """ Inverse operation of mu-law transform for 16-bit integers. """
    assert y.min() >= -1 and y.max() <= 1
    return torch.sign(y) / 32768. * ((1 + 32768.) ** torch.abs(y) - 1)


def weights_init(m, gain=None):
    """ Applies orthogonal initialization to dense weights. """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if gain is None:
            gain = (0.5 * m.weight.size(1)) ** -0.5
        nn.init.orthogonal_(m.weight, gain=gain)


def print_list_values(*args, prefix="\t"):
    """ Prints list of tensors, rounded to 4 decimal points. """
    args = [round(arg.cpu().item(), 4) for arg in args]
    print(prefix, *args)


def print_batch_stats(batch, prefix="\t"):
    """ Prints the min, max, mean, and std of a batch tensor. """
    batch_min = round(batch.min().cpu().item(), 4)
    batch_max = round(batch.max().cpu().item(), 4)
    batch_mean = round(batch.mean().cpu().item(), 4)
    batch_std = round(batch.std().cpu().item(), 4)
    print(prefix, batch_min, batch_max, batch_mean, batch_std)


def print_cuda_memory(gpu: int = 0):
    """ Prints current memory stats of gpu. """
    t = torch.cuda.get_device_properties(gpu).total_memory
    c = torch.cuda.memory_cached(gpu)
    a = torch.cuda.memory_allocated(gpu)

    print("GPU {}".format(gpu))
    print("\tTotal memory: {}".format(t))
    print("\tCached memory: {}".format(c))
    print("\tAllocated memory: {}".format(a))

