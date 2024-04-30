from pocketsphinx import AudioFile, Pocketsphinx
from FeatureExtractions import *
from files import load_ship_data, segment_audio_data, create_segment_dict
import tempfile
import os
import soundfile as sf
def transcribe_segment(entry):
    
    # Create a NamedTemporaryFile for writing
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    try:
        audio_data = entry['audio']
        sample_rate = entry['sample_rate']
        
        # Write the audio data to the temporary file
        sf.write(temp_file.name, audio_data, sample_rate)
        
        # Initialize a Pocketsphinx object
        ps = Pocketsphinx()
        
        # Decode the audio file
        ps.decode(temp_file.name)
        
        # Fetch the hypothesis (the recognized text)
        return ps.hypothesis()

    finally:
        # Close and remove the temporary file
        temp_file.close()
        os.unlink(temp_file.name)


def segment_theaudio_data(data, segment_length=5, overlap_percent=0.25):
    segmented_data = []
    for entry in data:
        print("entry: ", entry)
        audio = entry['audio']
        sr = entry['sample_rate']
        segment_length_samples = int(sr * segment_length)
        overlap_samples = int(segment_length_samples * overlap_percent)
        step = segment_length_samples - overlap_samples

        for start in range(0, len(audio), step):
            print("start: ", start)
            end = start + segment_length_samples
            segment = audio[start:end]
            if len(segment) < segment_length_samples:
                segment = np.pad(segment, (0, segment_length_samples - len(segment)), mode='constant', constant_values=(0, 0))

            # Create segment dict including transcription
            segment_dict = create_segment_dict(entry, sr, segment)
            # Transcribe the audio segment
            segment_dict['transcription'] = transcribe_segment(segment_dict)
            segmented_data.append(segment_dict)

    return segmented_data


dataset = load_ship_data()
segmented_dataset = segment_theaudio_data(dataset, segment_length=5, overlap_percent=0.25)

# Output results
for segment in segmented_dataset:
    print(segment['transcription'])