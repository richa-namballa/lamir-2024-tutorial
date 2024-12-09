import os
import torch
from torchaudio.transforms import InverseSpectrogram
import matplotlib.pyplot as plt


def plot_loss(history, save_path='./loss.png'):
    """
    Plot loss curves.

    Args:
        history (dict): training and validation history
        save_path (str, optional): where to save plot PNG file
    """
    # generate a sequence of integers for epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # plot
    plt.plot(epochs, history['train_loss'], label="Training Loss")
    plt.plot(epochs, history['val_loss'], label="Validation Loss")

    # make it pretty
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')

    # save plot
    plt.savefig(save_path, bbox_inches='tight')


class Spec2Audio():
    def __init__(self, n_fft, hop_size, device):
        """Convert a spectrogram to audio.

        Args:
            n_fft (int): FFT size
            hop_size (int): length of hop between windows
            device (torch.device): device to allocate data to
        """
        self.n_fft = n_fft
        self.device = device
        self.ifft = InverseSpectrogram(n_fft=n_fft,
                                       hop_length=hop_size)
        self.ifft = self.ifft.to(device)

    def pad_spec(self, spec):
        """Zero-pad spectrogram.

        Args:
            spec (torch.Tensor): spectrogram representation of
                                 shape (N, C, F, T) where
                                 N is batch size,
                                 C is the number of channels,
                                 F is the number of frequency bins, and
                                 T is the number of FFT time frames
        Returns:
            spec (torch.Tensor): spectrogram
        """
        # compute the expected number of frequency bins
        expected_bins = self.n_fft // 2 + 1

        # unpack tensor size
        batch, chan, freq, frames = spec.size()

        # if there are missing frequency bins
        if freq < expected_bins:
            pad_size = expected_bins - freq
            # zero pad
            padding = torch.zeros((batch, chan, pad_size, frames))
            padding = padding.to(self.device)
            spec = torch.concat((spec, padding), dim=2)

        # if there are extra frequency bins
        if freq > expected_bins:
            # remove the extras
            spec = spec[:, :, :expected_bins, :]

        return spec

    def to_audio(self, spec, phase):
        """Convert a spectrogram to audio.

        Args:
            spec (torch.Tensor): magnitude spectrogram representation of
                                 shape (N, F, T) where
                                 N is the batch size,
                                 F is the number of frequency bins, and
                                 T is the number of FFT time frames
            phase (torch.Tensor): complex spectrogram phase representation
                                  of shape (N, F, T)

        Returns:
            torch.Tensor: waveform representation of
                          shape (N, S) where
                          N is the batch size, and
                          S is the number of samples
        """
        # apply original mix phase to magnitude spectrogram
        complex_spec = spec * phase

        # pad spectrogram
        complex_spec = self.pad_spec(complex_spec)

        # perform inverse Fourier transform
        # remove channel dimension of 1
        waveform = self.ifft(torch.squeeze(complex_spec))

        return waveform
