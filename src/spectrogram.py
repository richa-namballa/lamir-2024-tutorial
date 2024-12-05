import os
import numpy as np
import librosa


def get_spec(audio, orig_rate, target_rate, fft_size, hop_size):
    """
    Compute the spectrogram of the audio.
    If the audio signal is stereophonic, it will
    be downsampled to monophonic.

    Args:
        audio (np.ndarray): audio signal
        orig_rate (int): sample rate of audio
        target_rate (int): sample rate to downsample to
        fft_size (int): number of FFT coefficients in the STFT
        hop_size (int): hop length between frames of the STFT

    Returns:
        (np.ndarray): complex-valued matrix of STFT coefficients
    """
    try:
        if len(audio.shape) > 1:
            # convert to mono
            if audio.shape[0] > audio.shape[1]:
                # (channels, samples)
                audio = audio.T
            audio = librosa.to_mono(audio)

        # downsample to reduce computational load
        y = librosa.resample(audio, orig_sr=orig_rate, target_sr=target_rate)

        # compute stft
        spec = librosa.stft(y, n_fft=fft_size, hop_length=hop_size)

        # remove last band
        b = spec.shape[0] - 1
        spec = spec[:b, :]

    except Exception as e:
        LOG.error(e)

    return spec


def process_file(file_path, target_rate, fft_size, hop_size, norm=True, mix=False):
    """
    Load audio and generate spectrogram.
    
    Args:
        file_path (str): path to audio file
        target_rate (int): sample rate to downsample to
        fft_size (int): number of FFT coefficients in the STFT
        hop_size (int): hop length between frames of the STFT
        norm (bool): whether to normalize the magnitude
        mix (bool): whether the audio file is the mix or not
    
    Returns:
        (np.ndarray, np.ndarray): spectrogram magnitude and phase
                                  if mix=False, phase is set to None
    """
    # load audio
    y, sr = librosa.load(file_path)

    # get spectrogram
    S = get_spec(y, sr, target_rate, fft_size, hop_size)

    # separate into magnitude and phase
    mag, phase = librosa.magphase(S)

    if norm:
        # normalize between 0 and 1
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

    # if the file is not a mixture, discard the phase component
    if not mix:
        phase = None

    # insert extra channel dimension to the mag spec
    mag = np.expand_dims(mag, axis=0)

    return mag, phase


def process_song(dir_path, save_path, target_source, target_rate, fft_size, hop_size):
    """
    Process an entire song.

    Args:
        dir_path (str): path to song directory
        save_path (str): .npz file path to save spectrograms to
        target_source (str): name of target instrument to generate
                             stem spectrogram of
        target_rate (int): sample rate to downsample to
        fft_size (int): number of FFT coefficients in the STFT
        hop_size (int): hop length between frames of the STFT
    """
    # process mixture
    mix_path = os.path.join(dir_path, "mixture.wav")
    mix_mag, mix_phase = process_file(mix_path, target_rate, fft_size, hop_size, mix=True)

    # process stem
    inst_path = os.path.join(dir_path, target_source + ".wav")
    stem_mag, _ = process_file(inst_path, target_rate, fft_size, hop_size)

    # save spectrograms to dictionary
    savez_dict = dict()
    savez_dict[target_source] = stem_mag
    savez_dict["mixture"] = mix_mag
    savez_dict["phase"] = mix_phase

    np.savez(save_path, **savez_dict)


def generate_spectrograms(source_dir, target_dir, target_source, target_rate, fft_size, hop_size):
    """
    Generate magnitude spectrograms from audio files in a directory.

    Args:
        source_dir (str): path to directory containing
                          audio stems in nested folders
        target_dir (str): path to directory where spectrograms
                          should be saved
        target_source (str): name of target instrument to generate
                             stem spectrogram of
        target_rate (int): sample rate to downsample to
        fft_size (int): number of FFT coefficients in the STFT
        hop_size (int): hop length between frames of the STFT
    """
    # make target dir if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # get all of the songs in the source dir
    songs = [s for s in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, s))]

    print("Generating spectrograms...")
    for s in songs:
        in_path = os.path.join(source_dir, s)
        out_path = os.path.join(target_dir, s + ".npz")

        process_song(in_path, out_path, target_source, target_rate, fft_size, hop_size)

    print(f"Spectrograms for {len(songs)} songs generated successfully!")
