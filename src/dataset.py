import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class SeparationDataset(Dataset):
    def __init__(self, dir_path, target_source, pct_files=1, patch_size=128, random_state=None):
        """
        Initialize a torch Dataset for loading a mixture and
        the target source stem.
        
        Args:
            dir_path (str): path to spectrogram data
            target_source (str): name of target source for separation
            pct_files (float): the percentage of files from the directory to use
            patch_size (int): the size of the random patch to extract
            random_state (int): random seed for reproducibility
        """
        self.dir_path = dir_path
        self.source = target_source
        self.patch_size = patch_size

        # get spectrogram file names
        all_songs = [s for s in os.listdir(dir_path) if s.endswith('.npz')]
        if random_state is not None:
            random.seed(random_state)
        num_files = int(pct_files * len(all_songs))
        self.songs = random.sample(all_songs, num_files)

        # load all of the spectrograms into memory
        # if OOM, change this to load spectrograms one-by-one
        self.load_songs()

    def load_songs(self):
        self.mixtures = []
        self.stems = []
        self.phi = []

        for s in self.songs:
            song_path = os.path.join(self.dir_path, s)
            song_dict = np.load(song_path)
            mix = song_dict["mixture"]
            stem = song_dict[self.source]
            phase = song_dict["phase"]

            if mix.shape != stem.shape:
                LOG.error("Mixture and source spectrograms must have the same shape!")
                raise ValueError

            self.mixtures.append(mix)
            self.stems.append(stem)
            self.phi.append(phase)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        mix = self.mixtures[idx]
        stem = self.stems[idx]
        phase = self.phi[idx]

        # insert extra channel dimension to the phase array
        phase = np.expand_dims(phase, axis=0)

        # get the number of frames in the spectrogram
        c, b, num_frames = mix.shape
        
        # randomly extract a patch of frames
        start_idx = np.random.randint(num_frames - self.patch_size)
        mix_patch = mix[:, :, start_idx:(start_idx + self.patch_size)]
        stem_patch = stem[:, :, start_idx:(start_idx + self.patch_size)]
        phase_patch = phase[:, :, start_idx:(start_idx + self.patch_size)]

        # return input, target, and phase
        return mix_patch, stem_patch, phase_patch


class TestDataset(Dataset):
    def __init__(self, dir_path, target_source, patch_size=128):
        """
        Initialize a torch Dataset for loading a mixture and
        the target source stem. This loads the entire audio as patches
        rather than randomly selecting a single patch per song.
        
        Args:
            dir_path (str): path to spectrogram data
            target_source (str): name of target source for separation
            patch_size (int): the size of the random patch to extract
        """
        self.dir_path = dir_path
        self.source = target_source
        self.patch_size = patch_size

        # get spectrogram file names
        self.songs = [s for s in os.listdir(dir_path) if s.endswith('.npz')]

        # load all of the spectrograms into memory
        # if OOM, change this to load spectrograms one-by-one
        self.load_songs()

    def load_songs(self):
        self.mixtures = []
        self.stems = []
        self.phi = []

        for s in self.songs:
            song_path = os.path.join(self.dir_path, s)
            song_dict = np.load(song_path)
            mix = song_dict["mixture"]
            stem = song_dict[self.source]
            phase = song_dict["phase"]

            if mix.shape != stem.shape:
                LOG.error("Mixture and source spectrograms must have the same shape!")
                raise ValueError

            self.mixtures.append(mix)
            self.stems.append(stem)
            self.phi.append(phase)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        mix = self.mixtures[idx]
        stem = self.stems[idx]
        phase = self.phi[idx]

        # insert extra channel dimension to the phase array
        phase = np.expand_dims(phase, axis=0)

        # get the number of frames in the spectrogram
        num_channels, num_bands, num_frames = mix.shape

        # compute the number of non-overlapping segments
        num_seg = num_frames // self.patch_size + 1
        
        # initialize arrays of patches
        mix_segments = np.zeros((num_seg, num_channels, num_bands, self.patch_size), dtype=np.float32)
        stem_segments = np.zeros((num_seg, num_channels, num_bands, self.patch_size), dtype=np.float32)
        phase_segments = np.zeros((num_seg, num_channels, num_bands, self.patch_size), dtype=np.complex64)

        for i in range(num_seg):
            # get start and end indicies for slicing
            start_idx = i * self.patch_size
            end_idx = min(start_idx + self.patch_size, num_frames)

            # find how much to slice in zero arrays
            if end_idx == num_frames:
                tmp = end_idx - start_idx
            else:
                tmp = self.patch_size

            # add patch to segment array
            mix_segments[i, :, :, :tmp] = mix[:, :, start_idx:end_idx]
            stem_segments[i, :, :, :tmp] = stem[:, :, start_idx:end_idx]
            phase_segments[i, :, :, :tmp] = phase[:, :, start_idx:end_idx]

        return mix_segments, stem_segments, phase_segments
