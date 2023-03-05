"""Voice Recognition AI project.

This module contains the data handeling script for the Voice Recognition AI project.

@Author: Emanuel-Ionut Otel
@Company: University of Southern Denmark
@Created: 2021-09-22
@Contact: emote21@student.sdu.dk
"""
#### ---- IMPORT AREA ---- ####
import os, torch, torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from audiomentations  import Compose, AddGaussianNoise, PitchShift, Shift
from torch.utils.data import Dataset
#### ---- IMPORT AREA ---- ####

#### ---- INIT AREA ---- ####
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir("..")
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),])
#### ---- INIT AREA ---- ####


class PodcastDataset(Dataset):
    """Podacast dataset class.
    
    Extends the torch.utils.data.Dataset class, for specific loading of the podcast dataset.
    
    :attr annotations_file:     The path to the annotations file.
    :attr audio_dir:            The path to the audio directory.
    :attr transformation:       The transformation to be applied to the audio samples.
    :attr target_sample_rate:   The target sample rate for the audio samples.
    :attr device:               The device to be used for the audio samples.
    :attr num_samples:          The number of samples for the audio samples.
    """
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, device, num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
        if transformation!=None:
            self.transformation = transformation.to(self.device)
        else:
            self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._white_noise(signal, sr)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        if(self.transformation != None):
            signal = self.transformation(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def _resample_if_necessary(self, signal, sr):
        """Resample the signal if the sample rate is different from the target sample rate.

        :param signal: The signal to be resampled.
        :param sr:     The sample rate of the signal.

        :return:       The resampled signal.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _right_pad_if_necessary(self, signal):
        """Right pad the signal if the number of samples is smaller than the target number of samples.
        
        :param signal: The signal to be padded.
        
        :return:       The padded signal."""
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal):
        """Cut the signal if the number of samples is larger than the target number of samples.

        :param signal: The signal to be cut.

        :return:       The cut signal.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        """Mix down the signal if the number of channels is larger than 1.

        :param signal: The signal to be mixed down.

        :return:       The mixed down signal.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _white_noise(self, signal, sr):
        """Add white noise to the signal.

        :param signal: The signal to be augmented.
        :param sr:     The sample rate of the signal.

        :return:       The augmented signal.
        """
        signal = augment(samples=signal, sample_rate=sr)
        return signal


if __name__ == "__main__":
    annotations_file = ".\dataset\data_set.csv"
    audio_dir = ".\dataset"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    sample_rate = 8000
    num_samples = sample_rate*5

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512,
                                                           n_mels=128)

    n_fft = 256
    win_length = None
    hop_length = 256
    n_mels = 512
    n_mfcc = 256

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk',
    }
                                                )

    pds = PodcastDataset(annotations_file=annotations_file, audio_dir=audio_dir, transformation=mfcc_transform,
                         target_sample_rate=sample_rate, device=device, num_samples=num_samples)

    signal, l1 = pds[4]
    signal2, l2 = pds[104]
    signal3, l3 = pds[204]

    signal = signal.to("cpu")
    signal2 = signal2.to("cpu")
    signal3 = signal3.to("cpu")

    print("Shape of spectrogram: {}".format(signal.size()))
    print("Shape of spectrogram: {}".format(signal2.size()))
    print("Shape of spectrogram: {}".format(signal3.size()))

    plt.figure()
    p = plt.imshow(signal.log2()[0, :, :].numpy())
    plt.title(l1)
    plt.figure()
    plt.imshow(signal2.log2()[0, :, :].numpy())
    plt.title(l2)
    plt.figure()
    plt.imshow(signal3.log2()[0, :, :].numpy())
    plt.title(l3)
    plt.show()
