import os
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
from spectrogram import process_file
from utils import Spec2Audio


def separate(mix_path, model, device, fft_size, hop_size, target_sr, patch_size=128):
    """Summary
    
    Args:
        mix_path (str): location of the mixture WAV file to separate
        model (nn.Module): PyTorch U-Net model
        device (torch.device): device to allocate model and data to
        target_sr (int): target sample rate to convert the audio to
        patch_size (int): size of spectrogram patch for model input
    
    Returns:
        np.ndarray: the separated stem
    """
    # initialize spectrograms to audio
    audio_converter = Spec2Audio(fft_size, hop_size, device)

    # send model to device
    model = model.to(device)

    # load mix and generate spectrogram
    print("Generating spectrogram...")
    mix_mag, mix_phase = process_file(mix_path, target_sr, fft_size, hop_size, norm=True, mix=True)

    # insert extra channel dimension to the phase array
    mix_phase = np.expand_dims(mix_phase, axis=0)

    # get dimensions of spectrogram
    print("Creating patches...")
    num_channels, num_bands, num_frames = mix_mag.shape

    # compute the number of non-overlapping segments
    num_seg = num_frames // patch_size

    # initialize arrays of patches
    mix_segments = np.zeros((num_seg, num_channels, num_bands, patch_size), dtype=np.float32)
    phase_segments = np.zeros((num_seg, num_channels, num_bands, patch_size), dtype=np.complex64)

    for i in range(num_seg):
        # get start and end indicies for slicing
        start_idx = i * patch_size
        end_idx = min(start_idx + patch_size, num_frames)

        # find how much to slice in zero arrays
        if end_idx == num_frames:
            tmp = end_idx - start_idx
        else:
            tmp = patch_size

        # add patch to segment array
        mix_segments[i, :, :, :tmp] = mix_mag[:, :, start_idx:end_idx]
        phase_segments[i, :, :, :tmp] = mix_phase[:, :, start_idx:end_idx]

    # convert to tensors
    mix_segments = torch.from_numpy(mix_segments).to(device)
    phase_segments = torch.from_numpy(phase_segments).to(device)

    # forward
    print("Running inference...")
    preds = model(mix_segments)

    # convert spectrograms to audio and move to CPU
    print("Converting stem to audio...")
    preds_audio = audio_converter.to_audio(preds, phase_segments).detach().cpu()

    # flatten along patch dimension
    y = torch.flatten(preds_audio)

    # normalize
    stem = y / torch.max(torch.abs(y))

    # convert to numpy array
    stem = stem.numpy()

    return stem
