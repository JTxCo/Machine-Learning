import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from files import load_ship_data, segment_audio_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Features extraction functions:
from FeatureExtractions import extract_mfccs, extract_all_mfccs


def find_minimum_audio_length(data):
    min_length = float('inf')  # Initialize with a very high number
    
    for entry in data:
        audio = entry['audio']
        if len(audio) < min_length:
            min_length = len(audio)
    
    return min_length

    return max_length

# loading the ship data without segmenting the audio data
dataset_long = load_ship_data()
print("Minimum audio length:", find_minimum_audio_length(dataset_long))
# Segmenting the audio data, overlapping it so that features are not missed
# Segment length of 5 seconds and overlap of 25% are defaults
dataset_segmented = segment_audio_data(dataset_long, segment_length=5, overlap_percent=0.25)
print("Minimum audio length after segmentation:", find_minimum_audio_length(dataset_segmented))

# Feature extraction
mfccs = extract_all_mfccs(dataset_segmented, sample_rate=22050, n_mfcc=13)

