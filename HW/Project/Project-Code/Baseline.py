import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# This is intended to find the baseline of performance for the model.
# Load an example audio file
file_path = '../DeepShip/Passengership/1.wav'
audio, sample_rate = librosa.load(file_path, sr=None)  # Load with the original sample rate

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
print("MFCCs shape:", mfccs.shape)

# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
