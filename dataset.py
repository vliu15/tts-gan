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

""" Dataset class for preprocessing and loading text and audio. """

from librosa.core import load
from librosa.util import normalize
import numpy as np
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from modules.mel import MelSpectrogram
from modules.phonemizer import cmudict, text_to_sequence


def files_to_list(filename):
    """ Takes a text file of filenames and makes a list of filenames. """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_metadata(filename):
    """ Loads transcriptions from metadata csv. """
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split("|")[:2] for line in f]

    return dict(filepaths_and_text)


class AudioDataset(Dataset):
    """ Dataset for handling audio io. """

    def __init__(self, audio_files: str, meta_file: str, cmudict_file: str, segment_length: int, length_scale: int, sampling_rate: int, augment: bool = False):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.length_scale = length_scale
        self.slice_length = segment_length // length_scale
        self.augment = augment

        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.metadata = load_metadata(meta_file)

        self.text_fn = cmudict.CMUDict(cmudict_file)

    @torch.no_grad()
    def __getitem__(self, index):
        # Process audio.
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        audio = audio[:audio.size(0) - audio.size(0) % self.length_scale]
        audio_len = audio.size(0)

        if audio.size(0) < self.segment_length:
            audio_offset = 0
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            )

        else:
            max_audio_offset = (audio.size(0) - self.segment_length) // self.length_scale  # grid offset to be multiples of length_scale (upsampling factor)
            audio_offset = random.randint(0, max_audio_offset) * self.length_scale
            audio = audio[audio_offset:audio_offset + self.segment_length]

        # Process text.
        text = self.metadata[os.path.splitext(os.path.basename(filename))[0]]
        text = text_to_sequence(text, ["english_cleaners"], self.text_fn)
        text = torch.LongTensor(text)
        text_len = text.size(0)

        # Audio len and offset are in downsampled resolutions.
        return text, text_len, audio, audio_len // self.length_scale, audio_offset // self.length_scale, self.slice_length

    def __len__(self):
        return len(self.audio_files)

    @staticmethod
    @torch.no_grad()
    def collate_fn(batch):
        # Unpack batch and compute lengths.
        text, text_len, audio, audio_len, audio_offset, slice_length = zip(*batch)
        x_len = torch.LongTensor(text_len)
        y_len = torch.LongTensor(audio_len)
        y_offset = torch.LongTensor(audio_offset)
        slice_length = torch.LongTensor(slice_length)

        # Create padded tensors.
        x = torch.LongTensor(len(batch), x_len.max())
        x.zero_()
        y = torch.stack(audio, dim=0)  # All audio loaded will be segments.

        # Populate padded tensors.
        for i in range(len(batch)):
            x[i, :len(text[i])] = text[i]
            y[i, :len(audio[i])] = audio[i]

        return x, x_len, y, y_len, y_offset, slice_length

    def load_wav_to_torch(self, full_path):
        """ Loads wavdata into torch array. """
        audio, sampling_rate = load(full_path, sr=self.sampling_rate)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            audio = audio * amplitude

        audio = torch.from_numpy(audio).float()
        return audio, sampling_rate
