import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt


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
# Using the pywavlets library to extract the wavelet packet decomposition
# defalt type is 
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



def plot_wavelet(wavelet_name='db1', family='db', level=5):
    # Create wavelet object
    wavelet = pywt.Wavelet(wavelet_name)
    
    # Get the wavelet function Psi at a given level of resolution
    phi, psi, x = wavelet.wavefun(level=level)
    
    # Plot the wavelet function and scaling function
    plt.figure(figsize=(12, 5))
    
    # Time domain representation
    plt.subplot(1, 2, 1)
    plt.plot(x, psi, label='Wavelet function (psi)')
    plt.plot(x, phi, label='Scaling function (phi)', linestyle='--')
    plt.title(f'Wavelet & Scaling Function in Time Domain: {family.upper()}{level}')
    plt.legend()
    
    # Frequency domain representation
    plt.subplot(1, 2, 2)
    freqs = np.fft.fftfreq(x.size, d=(x[1] - x[0]))
    fft_wavelet = np.fft.fft(psi)
    plt.plot(freqs[:freqs.size // 2], 20 * np.log10(np.abs(fft_wavelet[:freqs.size // 2])), label='Magnitude Spectrum (dB)')
    plt.title('Magnitude Spectrum in Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_wavelet('db1', 'db', level=5)




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

