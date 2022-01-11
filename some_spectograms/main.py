import torch
import torchaudio
import librosa
import torchaudio.functional as F
import torchaudio.transforms as T
import os
from pytorchway import plot_waveform, print_stats, play_audio, plot_spectrogram, get_rir_sample, plot_pitch, plot_kaldi_pitch
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import requests
import math

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
_SAMPLE_DIR = "_sample_data"

SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "esben05.wav")
SAMPLE_WAV_PATH2 = os.path.join(_SAMPLE_DIR, "peter05.wav")

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
waveform2, sample_rate2 = torchaudio.load(SAMPLE_WAV_PATH2)

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(waveform)
plot_spectrogram(melspec[0], title="Esben", ylabel='mel freq')

melspec = mel_spectrogram(waveform2)
plot_spectrogram(melspec[0], title="Peter", ylabel='mel freq')

SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "esben05 (1).wav")
SAMPLE_WAV_PATH2 = os.path.join(_SAMPLE_DIR, "peter05 (1).wav")

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
waveform2, sample_rate2 = torchaudio.load(SAMPLE_WAV_PATH2)

melspec = mel_spectrogram(waveform)
plot_spectrogram(melspec[0], title="Esben", ylabel='mel freq')

melspec = mel_spectrogram(waveform2)
plot_spectrogram(melspec[0], title="Peter", ylabel='mel freq')


n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'hop_length': hop_length,
      'mel_scale': 'htk',
    }
)

mfcc = mfcc_transform(waveform)
plot_spectrogram(mfcc[0])

mfcc = mfcc_transform(waveform2)
plot_spectrogram(mfcc[0])

SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "esben05.wav")
SAMPLE_WAV_PATH2 = os.path.join(_SAMPLE_DIR, "peter05.wav")

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
waveform2, sample_rate2 = torchaudio.load(SAMPLE_WAV_PATH2)

mfcc = mfcc_transform(waveform)
plot_spectrogram(mfcc[0])

mfcc = mfcc_transform(waveform2)
plot_spectrogram(mfcc[0])

plt.close('all')

pitch = F.detect_pitch_frequency(waveform, sample_rate)
pitch2 = F.detect_pitch_frequency(waveform2, sample_rate2)
plot_pitch(waveform, sample_rate, pitch, "Esben")
plot_pitch(waveform2, sample_rate2, pitch2, "Peter")

SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "esben05 (2).wav")
SAMPLE_WAV_PATH2 = os.path.join(_SAMPLE_DIR, "peter05 (2).wav")

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
waveform2, sample_rate2 = torchaudio.load(SAMPLE_WAV_PATH2)

pitch = F.detect_pitch_frequency(waveform, sample_rate)
pitch2 = F.detect_pitch_frequency(waveform2, sample_rate2)
plot_pitch(waveform, sample_rate, pitch, "Esben 02")
plot_pitch(waveform2, sample_rate2, pitch2, "Peter 02")

input("Press Enter to continue...")