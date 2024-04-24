from pathlib import Path
import pandas as pd
import librosa
import numpy as np

def load_ship_data():
    
    # Specify the base directory containing ship data folders
    base_directory = Path('../DeepShip/')

    # Initialize a list to store data
    dataset = []

    # Iterate through each subdirectory in the base directory
    for dir_entry in base_directory.iterdir():
        if dir_entry.is_dir():  # Check if it's a directory
            print(f"Directory: {dir_entry}")
            metafile_path = dir_entry / f'{dir_entry.name.lower()}-metafile'
            
            # Read metafile data into a DataFrame
            if metafile_path.exists():
                meta_df = pd.read_csv(metafile_path, index_col=False)
                # Drop the unnecessary columns, if it exists
                if 'Unnamed: 7' in meta_df.columns:
                    meta_df = meta_df.drop(columns=['Unnamed: 7'])
                # Assign the correct column names
                meta_df.columns = ['ID', 'Class_ID', 'Ship_Name', 'Date', 'Time', 'Duration', 'Distance']
    
                # Iterate through the metafile DataFrame and process corresponding audio files
                for index, row in meta_df.iterrows():
                    # print(row)
                    audio_path = dir_entry / f"{row['ID']}.wav"
                    # Check if audio file exists
                    if audio_path.exists():
                        audio, sr = librosa.load(audio_path, sr=None)  # Load the audio file
                        # Append loaded audio data with metadata to the dataset
                        dataset.append({
                            'audio': audio,
                            'sample_rate': sr,
                            'class_id': row['Class_ID'],
                            'ship_name': row['Ship_Name'],
                            'date': row['Date'],
                            'time': row['Time'],
                            'duration': row['Duration'],
                            'distance': row['Distance']
                        })
                # Print status
                print(f"Processed {len(meta_df)} entries from {metafile_path}")
            else:
                print(f"Metafile not found for directory: {dir_entry.name}")
        else:
            print(f"Skipping file: {dir_entry}")  # Not a directory, skip processing
        print("dataset length at this point: ", len(dataset))
    return dataset

    
    
    # Function to to segment the audio data into fixed length segments
    # I am taking the dataframe in and then breaking up the individual audio files into X length segments
    # I will be optimizing the segment length so this is why it is being done after the data is loaded
    
def segment_audio_data(data, segment_length=5, overlap_percent=0.25):
    segmented_data = []
    for entry in data:
        audio = entry['audio']
        sr = entry['sample_rate']
        segment_length_samples = int(sr * segment_length)
        overlap_samples = int(segment_length_samples * overlap_percent)
        step = segment_length_samples - overlap_samples

        # Processing all segments with the specified overlap
        for start in range(0, len(audio), step):
            end = start + segment_length_samples
            segment = audio[start:end]
            if len(segment) < segment_length_samples:
                segment = np.pad(segment, (0, segment_length_samples - len(segment)), mode='constant', constant_values=(0, 0))
            segmented_data.append(create_segment_dict(entry, sr, segment))

    return segmented_data

def create_segment_dict(entry, sr, segment):
    # Helper function to create a dictionary for each segment and avoid code duplication.
    return {
        'audio': segment,
        'sample_rate': sr,
        'class_id': entry['class_id'],
        'ship_name': entry['ship_name'],
        'date': entry['date'],
        'time': entry['time'],
        'duration': entry['duration'],
        'distance': entry['distance']
    }
