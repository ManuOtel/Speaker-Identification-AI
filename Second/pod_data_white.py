import os
import pandas as pd
import numpy as np
from audiomentations import LowPassFilter, Compose, AddGaussianSNR, AddGaussianNoise, TimeStretch, PitchShift, Shift
import torchaudio
import torch
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

augment = Compose([
    AddGaussianSNR(p=0),
    PitchShift(min_semitones=-1, max_semitones=1, p=0),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    #LowPassFilter(min_cutoff_freq=300, max_cutoff_freq=500, p=1),
])


class PodcastDataset(Dataset):

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

    # len(pd)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = signal.to("cpu")
        signal = signal.cpu().detach().numpy()
        #signal = np.reshape(signal, (40000, ))
        signal = self._white_noise(signal, self.target_sample_rate)
        #signal = np.reshape(signal, (1, 40000))
        signal = torch.from_numpy(signal)
        signal = signal.to(self.device)
        if(self.transformation != None):
            signal = self.transformation(signal)
        return signal, label

    # list[1] -> list.__getitem__(1)

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _white_noise(self, signal, sr):
        signal = augment(samples=signal, sample_rate=sr)
        return signal


if __name__ == "__main__":
    annotations_file = "D:\Manu\SDU\Projects\DNN\Data\dataset\data_set.csv"
    audio_dir = "D:\Manu\SDU\Projects\DNN\Data\dataset"

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
