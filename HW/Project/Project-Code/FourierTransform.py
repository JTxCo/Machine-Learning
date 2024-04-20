import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt

# File for Fourier Transform on .wav files


audio_file_path = '../DeepShip/Passengership/4.wav'


y, sr = librosa.load(audio_file_path, sr=None) 
D = librosa.stft(y)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to decibels (dB)

# Plot Spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency (log scale)')
plt.show()