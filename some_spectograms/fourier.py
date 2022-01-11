import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

esben1 = 'esben05.wav'
e1, esr1 = librosa.load(esben1)
esben2 = 'esben05 (6).wav'
e2, esr2 = librosa.load(esben2)
esben3 = 'esben05 (2).wav'
e3, esr3 = librosa.load(esben3)
esben4 = 'esben05 (3).wav'
e4, esr4 = librosa.load(esben4)
esben5 = 'esben05 (4).wav'
e5, esr5 = librosa.load(esben5)
esben6 = 'esben05 (5).wav'
e6, esr6 = librosa.load(esben6)

peter1 = 'peter05.wav'
p1, psr1 = librosa.load(peter1)
peter2 = 'peter05 (1).wav'
p2, psr2 = librosa.load(peter2)
peter3 = 'peter05 (2).wav'
p3, psr3 = librosa.load(peter3)
peter4 = 'peter05 (3).wav'
p4, psr4 = librosa.load(peter4)
peter5 = 'peter05 (4).wav'
p5, psr5 = librosa.load(peter5)
peter6 = 'peter05 (5).wav'
p6, psr6 = librosa.load(peter6)

plt.figure()
n_fft = 512
fe1 = np.abs(librosa.stft(e1[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fe2 = np.abs(librosa.stft(e2[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fe3 = np.abs(librosa.stft(e3[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fe4 = np.abs(librosa.stft(e4[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fe5 = np.abs(librosa.stft(e5[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fe6 = np.abs(librosa.stft(e6[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

fp1 = np.abs(librosa.stft(p1[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fp2 = np.abs(librosa.stft(p2[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fp3 = np.abs(librosa.stft(p3[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fp4 = np.abs(librosa.stft(p4[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fp5 = np.abs(librosa.stft(p5[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
fp6 = np.abs(librosa.stft(p6[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

plt.subplot(261)
plt.plot(fe1)
plt.subplot(262)
plt.plot(fe2)
plt.subplot(263)
plt.plot(fe3)
plt.subplot(264)
plt.plot(fe4)
plt.subplot(265)
plt.plot(fe5)
plt.subplot(266)
plt.plot(fe6)

plt.subplot(267)
plt.plot(fp1)
plt.subplot(268)
plt.plot(fp2)
plt.subplot(2, 6, 9)
plt.plot(fp3)
plt.subplot(2, 6, 10)
plt.plot(fp4)
plt.subplot(2, 6, 11)
plt.plot(fp5)
plt.subplot(2, 6, 12)
plt.plot(fp6)

plt.show()

# trim silent edges
#esben, _ = librosa.effects.trim(y)
#peter, _ = librosa.effects.trim(x)

#plt.figure()
#plt.subplot(251)
#librosa.display.waveplot(esben, sr=sr1)
#plt.subplot(252)
#librosa.display.waveplot(peter, sr=sr2)