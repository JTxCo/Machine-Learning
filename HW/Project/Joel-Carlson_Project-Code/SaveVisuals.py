import matplotlib.pyplot as plt
import os
import numpy as np

def save_figure(figure, filename, directory='../Visuals'):
    """
    Saves a Matplotlib figure to a given directory.

    Args:
        figure (Figure): The Matplotlib figure to save.
        filename (str): The desired filename (without extension).
        directory (str): The directory to save the figure in.

    Returns:
        str: The full path of the saved file.
    """
    # Make sure the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define full path
    full_path = os.path.join(directory, filename)

    # Save figure
    figure.savefig(f"{full_path}.png", dpi=300, bbox_inches='tight')

    return full_path

def plot_audio(audio, sample_rate):
    """
    Plots the audio signal in time domain.

    Args:
        audio (np.array): Audio signal.
        sample_rate (int): Sample rate of the audio signal.
    """
    # Create an array of time points in seconds
    duration = len(audio) / sample_rate
    time = np.linspace(0., duration, len(audio))

    # Create a new figure and plot the audio signal
    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='Time (s)', ylabel='Amplitude',
           title='Time Domain Plot of Audio Signal')
    ax.grid()
    
    plt.show()

    return fig



    