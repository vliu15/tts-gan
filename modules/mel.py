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

""" Contains short-time Fourier transform and log-mel spectrogram modules. """

import librosa
from librosa.util import pad_center, tiny
import numpy as np
import random
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogram(nn.Module):
    """ Computes mel-spectrogram for segments of audio data.

    Scale and shift is applied to mel before taking the log, as in (Donahue et al. 2020)
    
    Args:
        filter_length: length of windowed signal after padding with zeros, number of rows in
            STFT matrix is (1 + filter_length // 2)
        hop_length: number of samples between adjacent STFT columns
        win_length: window size to apply to audio (before padding with zeros to filter_length)
        n_mels: number of mel bands to generate
        sampling_rate: sampling rate of the audio
        mel_fmin: lowest frequency in Hz
        mel_fmax: highest frequency in Hz
        window: name of the window to apply to audio

    References:
    > (Donahue et al. 2020) End-to-End Adversarial Text-to-Speech, https://arxiv.org/abs/2006.03575
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = None,
        n_mels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = None,
        window: str = "hann",
    ):
        super().__init__()
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def forward(self, audio, jitter_steps: int = 0):
        assert audio.min() >= -1 and audio.max() <= 1
        # Add batch dimension if not already present.
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        # Phase shift by jitter_steps if specified.
        if jitter_steps > 0:
            length = audio.size(-1)
            audio = F.pad(audio, (jitter_steps, jitter_steps), "constant")
            random_start = random.randint(0, 2 * jitter_steps)
            audio = audio[:, random_start:random_start + length]

        # Full-precision for stft and log-mel computation.
        magnitudes = self.stft.transform(audio)
        mel = torch.matmul(self.mel_basis, magnitudes)
        mel = torch.log(1 + 10000 * mel)  # shift and scale mel according to EATS.

        return mel

    def mel_len(self, audio_len):
        return audio_len // self.hop_length


class STFT(nn.Module):
    """ Performs short-time fourier transform on segments of audio data.
    
    Args:
        filter_length: length of windowed signal after padding with zeros, number of rows in
            STFT matrix is (1 + filter_length // 2)
        hop_length: number of samples between adjacent STFT columns
        win_length: window size to apply to audio (before padding with zeros to filter_length)
        window: name of the window to apply to audio
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = None,
        window: str = "hann",
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = (self.filter_length - self.hop_length) // 2
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert filter_length >= self.win_length
        # Get window and zero center pad it to filter_length.
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # Window the bases.
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)

    def transform(self, input_data):
        """ Applies short-time Fourier transform. """
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # Similar to librosa, reflect-pad the input.
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode="reflect")
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        return magnitude

    def inverse(self, magnitude, phase):
        """ Applies inverse short-time Fourier transform. """
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = librosa.filters.window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )

            # Remove modulation effects.
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # Scale by hop ratio.
            inverse_transform *= self.filter_length / self.hop_length

        inverse_transform = inverse_transform[:, :, self.pad_amount :]
        inverse_transform = inverse_transform[:, :, : -self.pad_amount :]

        return inverse_transform
