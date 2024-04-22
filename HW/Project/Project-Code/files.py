from pathlib import Path
import pandas as pd
import librosa


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