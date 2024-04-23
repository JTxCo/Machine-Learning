import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt


'''
 These are functions that do the feature extraction for the audio data
 I am going to be using the librosa library to extract the features from the audio data
 Extracting:
    1.	Wavelet packet decomposition 
    2.	Linear Predictive Coding (LPC) 
    3.	Fisher Criterion for feature selection 
    4.	Cepstrum, Mel spectrogram, MFCC, Constant Q Transform (CQT), Gammatone Frequency Cepstral Coefficients (GFCC), and additional Wavelet packets.

 
'''
# Wavelet Packet Decomposition









def extract_mfccs(audio, sample_rate, n_mfcc=13):
    """
    Extract MFCCs from an audio signal.

    Args:
        audio (np.array): The audio signal from which to extract features.
        sample_rate (int): The sample rate of the audio signal.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        np.array: Numpy array of MFCCs.
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return mfccs
    except Exception as e:
        print(f"Failed to extract MFCCs: {e}")
        return None


def extract_all_mfccs(entries, sample_rate, n_mfcc=13):
    """
    Extracts MFCCs from all entries in a given list or dataset.

    Args:
        entries (list): A list of dicts, each with an 'audio' key containing the audio data.
        sample_rate (int): The sample rate to be used for all audio data.
        n_mfcc (int): Number of MFCCs to return.

    Returns:
        list: List of MFCC arrays.
    """
    mfccs = []
    for entry in entries:
        audio = entry['audio']
        mfcc_result = extract_mfccs(audio, sample_rate, n_mfcc)
        if mfcc_result is not None:
            mfccs.append(mfcc_result)
        else:
            print(f"Skipping entry due to MFCC extraction error.")
    return mfccs

