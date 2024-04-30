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
from SaveVisuals import *

dataset_long = load_ship_data()
dataset_segmented = segment_audio_data(dataset_long, segment_length=5, overlap_percent=0.25)
audio_clip = dataset_segmented[0]['audio']
sample_rate = dataset_segmented[0]['sample_rate']

fig = plot_audio(audio_clip, sample_rate)
save_figure(fig, 'audio_plot_segmented', directory='../Visuals')


# dataset_segmented = segment_audio_data(dataset_long, segment_length=5, overlap_percent=0.25)
# print("Length of dataset_long:", len(dataset_segmented))
# labels = [entry['class_id'] for entry in dataset_segmented]
# lpc_coeffs_all = [lpc(sample, order=13) for sample in dataset_segmented]
# lpc_coeffs_clean = [coeff for coeff in lpc_coeffs_all if coeff is not None]
# labels_clean = [labels[i] for i, coeff in enumerate(lpc_coeffs_all) if coeff is not None]



# # Info gain testing
# info_gains = info_gain(lpc_coeffs_clean, labels_clean)
# fig = plot_info_gain(info_gains)

# # Save the figure
# save_figure(fig, 'info_gain_LPC', directory='../Visuals')

