import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

esben1 = 'esben05.wav'
E1, esr1 = librosa.load(esben1)
esben2 = 'esben05 (6).wav'
E2, esr2 = librosa.load(esben2)
esben3 = 'esben05 (2).wav'
E3, esr3 = librosa.load(esben3)
esben4 = 'esben05 (3).wav'
E4, esr4 = librosa.load(esben4)
esben5 = 'esben05 (4).wav'
E5, esr5 = librosa.load(esben5)
esben6 = 'esben05 (5).wav'
E6, esr6 = librosa.load(esben6)

peter1 = 'peter05.wav'
P1, psr1 = librosa.load(peter1)
peter2 = 'peter05 (1).wav'
P2, psr2 = librosa.load(peter2)
peter3 = 'peter05 (2).wav'
P3, psr3 = librosa.load(peter3)
peter4 = 'peter05 (3).wav'
P4, psr4 = librosa.load(peter4)
peter5 = 'peter05 (4).wav'
P5, psr5 = librosa.load(peter5)
peter6 = 'peter05 (5).wav'
P6, psr6 = librosa.load(peter6)

n_fft = 512
hop_length = 512

plt.figure()

plt.subplot(231)
E1 = np.abs(librosa.stft(E1, n_fft=n_fft,  hop_length=hop_length))
E1B = librosa.amplitude_to_db(E1, ref=np.max)
librosa.display.specshow(E1B, sr=esr1, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(232)
E2 = np.abs(librosa.stft(E2, n_fft=n_fft,  hop_length=hop_length))
E2B = librosa.amplitude_to_db(E2, ref=np.max)
librosa.display.specshow(E2B, sr=esr2, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(233)
E3 = np.abs(librosa.stft(E3, n_fft=n_fft,  hop_length=hop_length))
E3B = librosa.amplitude_to_db(E3, ref=np.max)
librosa.display.specshow(E3B, sr=esr3, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(234)
E4 = np.abs(librosa.stft(E4, n_fft=n_fft,  hop_length=hop_length))
E4B = librosa.amplitude_to_db(E4, ref=np.max)
librosa.display.specshow(E4B, sr=esr4, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(235)
E5 = np.abs(librosa.stft(E5, n_fft=n_fft,  hop_length=hop_length))
E5B = librosa.amplitude_to_db(E5, ref=np.max)
librosa.display.specshow(E5B, sr=esr5, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(236)
E6 = np.abs(librosa.stft(E6, n_fft=n_fft,  hop_length=hop_length))
E6B = librosa.amplitude_to_db(E6, ref=np.max)
librosa.display.specshow(E6B, sr=esr6, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')


plt.figure()

plt.subplot(231)
P1 = np.abs(librosa.stft(P1, n_fft=n_fft,  hop_length=hop_length))
P1B = librosa.amplitude_to_db(P1, ref=np.max)
librosa.display.specshow(P1B, sr=psr1, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(232)
P2 = np.abs(librosa.stft(P2, n_fft=n_fft,  hop_length=hop_length))
P2B = librosa.amplitude_to_db(P2, ref=np.max)
librosa.display.specshow(P2B, sr=psr2, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(233)
P3 = np.abs(librosa.stft(P3, n_fft=n_fft,  hop_length=hop_length))
P3B = librosa.amplitude_to_db(P3, ref=np.max)
librosa.display.specshow(P3B, sr=psr3, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(234)
P4 = np.abs(librosa.stft(P4, n_fft=n_fft,  hop_length=hop_length))
P4B = librosa.amplitude_to_db(P4, ref=np.max)
librosa.display.specshow(P4B, sr=psr4, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(235)
P5 = np.abs(librosa.stft(P5, n_fft=n_fft,  hop_length=hop_length))
P5B = librosa.amplitude_to_db(P5, ref=np.max)
librosa.display.specshow(P5B, sr=psr5, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(236)
P6 = np.abs(librosa.stft(P6, n_fft=n_fft,  hop_length=hop_length))
P6B = librosa.amplitude_to_db(P6, ref=np.max)
librosa.display.specshow(P6B, sr=psr6, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')


plt.show()