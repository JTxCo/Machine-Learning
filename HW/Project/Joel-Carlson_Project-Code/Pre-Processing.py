import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from files import load_ship_data, segment_audio_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Features extraction functions:
from FeatureExtractions import *


def find_minimum_audio_length(data):
    min_length = float('inf')  # Initialize with a very high number
    
    for entry in data:
        audio = entry['audio']
        if len(audio) < min_length:
            min_length = len(audio)
    
    return min_length


# loading the ship data without segmenting the audio data
dataset_long = load_ship_data()
print("Minimum audio length:", find_minimum_audio_length(dataset_long))
# original length:
# print("Length of dataset_long:", len(dataset_long))
# Segmenting the audio data, overlapping it so that features are not missed
# Segment length of 5 seconds and overlap of 25% are defaults
dataset_segmented = segment_audio_data(dataset_long, segment_length=5, overlap_percent=0.25)
print("Minimum audio length after segmentation:", find_minimum_audio_length(dataset_segmented))

# sample rate:
print("Sample rate of dataset_segmented:", dataset_segmented[0]['sample_rate'])
# length:
# print("Length of dataset_long:", len(dataset_segmented))
# Feature extraction
mfccs = extract_all_mfccs(dataset_segmented, sample_rate=22050, n_mfcc=13)

# wavelet visualization:
plot_audio_wavelet(dataset_segmented[0]['audio'], dataset_segmented[0]['sample_rate'],wavelet_name='gaus1', max_level=100)



# testing new feature extraction functions:
# waveletpacket decomposition:
wavelet_packet = wavelet_packet_decomposition(dataset_segmented[0]['audio'], wavelet='db1', level=5)
test_wavelet_output(wavelet_packet)

# Get the continuous wavelet transform
coef, freqs = continuous_wavelet_transform(dataset_segmented[0]['audio'],dataset_segmented[0]['sample_rate'], wavelet='morl', max_scale=10)
# Test the output
test_continuous_transform_output(coef, freqs)


# testing discrete wavelet transform:
coeffs = discrete_wavelet_transform(dataset_segmented[0]['audio'], wavelet='db1')
test_discrete_transform_output(coeffs)


# working with LPC
lpc_coeffs = lpc(dataset_segmented[0], order=13)
display_lpc(lpc_coeffs)