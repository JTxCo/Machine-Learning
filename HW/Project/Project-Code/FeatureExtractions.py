import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
import pandas as pd

'''
 These are functions that do the feature extraction for the audio data
 I am going to be using the librosa library to extract the features from the audio data
 

 follows format:      
        A. Function
        B. Testing with output
        
        
 Extracting:
    1.	Continuous Wavelet Transform (CWT)
    2.  Discrete Wavelet Transform (DWT)
    3.  Wavelet packet decomposition (WPD)
    4.	Linear Predictive Coding (LPC) 
    5.	Fisher Criterion for feature selection 
    6.	Cepstrum, Mel spectrogram, MFCC, Constant Q Transform (CQT), Gammatone Frequency Cepstral Coefficients (GFCC), and additional Wavelet packets.

 
'''
def continuous_wavelet_transform(audio, sample_rate, wavelet='morl', max_scale= 100):
    '''
        Extract continuous wavelet transform from the audio data.
        
        Args:
            audio (np.array): The input audio signal.
            sample_rate (int): The sample rate of the audio signal.
            wavelet (str): The type of wavelet to use, set to 'morl' for the Morlet wavelet.
            max_scale (int): The maximum scale to use for the wavelet transform.
            
        Returns:
            np.array: Continuous wavelet transform coefficients.
            np.array: Corresponding frequency array for scales.
    '''
    # Ensure scale starts at 1 to avoid zero or negative scale values and goes up to max_scale
    scales = np.arange(1, max_scale + 1)
    coef, freqs = pywt.cwt(audio, scales, wavelet, sampling_period=1/sample_rate)
    
    return coef, freqs


def test_continuous_transform_output(coef, freqs, max_scale = 10):
    # Print shape of the coefficients
    print("Shape of coefficients:", coef.shape)
    
    # Print some coefficients
    print("Some Coefficients:\n", coef[:10, :10])  # First 10 scales and 10 time points - adjust as necessary
    
    # Print some frequencies 
    print("Some Frequencies:\n", freqs[:10])  # First 10 scales - adjust as necessary
    
    # If you'd like, you could plot the CWT result as a 2D heatmap:
    plt.figure(figsize=(10,10))
    plt.imshow(coef, aspect='auto', cmap='coolwarm', extent=[0, len(coef[0]), 1, max_scale])
    plt.colorbar(label='CWT Coefficient')
    plt.show()
    
    # Or you could plot a single scale - here's how you might plot the first scale:
    plt.figure(figsize=(10,4))
    plt.plot(coef[0])
    plt.title("CWT Coefficients - Scale 1")
    plt.show()

def discrete_wavelet_transform(audio, wavelet='db1', level=None):
    '''
        Extracting the discrete wavelet transform (DWT) from the audio data.
        
        Args:
            audio (np.array): The audio signal from which to extract features.
            wavelet (str): The wavelet to use for decomposition, typically set to 'db1' for the Haar wavelet.
            level (int): Optional. The maximum level of decomposition to perform. Defaults to the maximum level possible if not specified.
                setting it to None will keep it at the max it can be. Otherwise less than max 
        Returns:
            list: List of wavelet coefficients for each level of decomposition.
            
    '''
    # If level is not set, calculate the maximum level possible for DWT
    if level is None:
        level = pywt.dwt_max_level(len(audio), pywt.Wavelet(wavelet))

    # Perform the DWT
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    
    return coeffs

def test_discrete_transform_output(coeffs):

    '''
        The plots show the Discrete Wavelet Transform (DWT) coefficients at each level of the decomposition. 
        The x-axis represents the indexed position in the wavelet coefficient array for each level
        The y-axis displays the coefficient values.

        Resolution/Sensitivity: 
            Level 1 shows the high-frequency details of the signal as it captures the shortest-scale information
            Level 2 +:  Higher decomposition levels (2, 3, etc.
    
        Magnitude of Coefficients: The magnitude of the wavelet coefficients at a specific index reflects how closely the scaled and shifted wavelet aligns with the audio signal at that point. 
        Larger magnitudes represent areas where the signal matches the wavelet shape more closely.
        
        Signal Details: High-frequency details (like noise or rapid changes) often appear in the first few levels of decomposition,
        while more significant structural components of the signal (like general shape or slower changes) become visible in the coefficients at higher levels.
    
    '''
    
    
    
    # Print the number of decomposition levels
    print("Decomposition Levels: ", len(coeffs))

    # Print some of the coefficients from each level
    for i, coeff in enumerate(coeffs):
        print(f"\nCoefficient details at level {i+1}:")
        print("-------------------------------")
        print("Length of Coefficient Array:", len(coeff))
        print("First Few Coefficients:\n", coeff[:10])  
        print("-------------------------------")

    # If you'd like, plot the coefficients at each level:
    for i, coeff in enumerate(coeffs):
        plt.figure(figsize=(10, 4))
        plt.plot(coeff)
        plt.title(f"DWT Coefficients - Level {i+1}")
        plt.show()



def wavelet_packet_decomposition(audio, wavelet='db1', level=5):
    '''
        Extracting the wavelet packet decomposition from the audio data
        Args:
            audio (np.array): The audio signal from which to extract features.
            wavelet (str): The wavelet to use for decomposition, set to db1 which is the Haar wavelet.
            level (int): The level of decomposition to perform.
            
        Returns:
            pywt.WaveletPacket: Wavelet packet decomposition of the audio signal.
            
    '''
    wp = pywt.WaveletPacket(data=audio, wavelet=wavelet, mode='symmetric', maxlevel=level)
    
    return {node.path : node.data for node in wp.get_level(level, 'natural')}


def test_wavelet_output(wavelet_packet):
    #  Length of the dictionary
    print("Number of nodes:", len(wavelet_packet))

    # Print some of the dictionary keys
    print("Some Keys:", list(wavelet_packet.keys())[:10])

    # Print first few values from a specific path
    print("Values from path 'aaaad':", wavelet_packet['aaaad'][:10])

    # for path in wavelet_packet.keys():
    #     plt.figure(figsize=(10,4))
    #     plt.title(f"Wavelet Packet Node: {path}")
    #     plt.plot(wavelet_packet[path])
    #     plt.show()

    summary_frame = pd.DataFrame(
        index=wavelet_packet.keys(),
        data={'min': [np.min(wavelet_packet[key]) for key in wavelet_packet.keys()],
            'max': [np.max(wavelet_packet[key]) for key in wavelet_packet.keys()],
            'mean': [np.mean(wavelet_packet[key]) for key in wavelet_packet.keys()],
            'std': [np.std(wavelet_packet[key]) for key in wavelet_packet.keys()]} 
    )

    print(summary_frame)

def plot_audio_wavelet(audio, sample_rate, wavelet_name='morl', max_level=None):
    max_level = max_level if max_level else sample_rate // 2
    scales = np.arange(1, max_level)

    coef, freqs = pywt.cwt(audio, scales, wavelet_name, sampling_period=1/sample_rate)
    plt.figure(figsize=(10, 6))
    log_coef = np.log(np.abs(coef) + 1e-9)
    plt.imshow(log_coef, extent=[0, len(audio) / sample_rate, freqs.min(), freqs.max()],
           cmap='PRGn', aspect='auto', vmax=log_coef.max(), vmin=log_coef.min())
    plt.yscale('log')
    plt.title('Wavelet Transform (scaleogram) of an Audio Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Magnitude')
    plt.show()






# Linear Predictive Coding (LPC) 
# Applied using this tutorial: https://www.kuniga.me/blog/2021/05/13/lpc-in-python.html

def lpc(data, order=13):
    '''
        Extract Linear Predictive Coding (LPC) coefficients from the audio data.
        
        Args:
            data (dict): A dictionary containing 'audio' and 'sample_rate' as keys.
            order (int): The order of the LPC model to use.
            
        Returns:
            np.array: LPC coefficients.
    '''
    audio = data['audio']
    sample_rate = data['sample_rate']
    
    # normalize the amplitude of the audio signal
    max_val = np.max(np.abs(audio))
    if max_val != 0:  # Safety check to prevent division by zero
        audio = audio / max_val
    
    # downsampling the audio signal
    target_sample_rate = 8000
    try:
        resampled_audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    except Exception as e:
        print(e)
        print('audio:', audio)
        print('sample_rate:', sample_rate)
        print('target_sample_rate:', target_sample_rate)
    
    # Apply LPC
    lpc_coeffs = librosa.lpc(y = audio, order = order)
    
    return lpc_coeffs


def display_lpc(lpc_coeffs):
    '''
        Display the LPC coefficients as a bar graph.
        
        Args:
            lpc_coeffs (np.array): The LPC coefficients to display.
    '''
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lpc_coeffs)), lpc_coeffs)
    plt.title("LPC Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.show()





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

