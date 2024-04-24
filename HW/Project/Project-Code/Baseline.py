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
# This is intended to find the baseline of performance for the model.
# This is going to be a simple CNN model that will be trained on the raw audio data.

# Load the ship data using files.py function
dataset = load_ship_data()


# Class ID corresponds to the ship class: 
# Cargo ships are 70s: 70, 71, 79, ...
# Tankers are 80s: 80, 81, 89, ...
# Passenger ships are 60s: 60, 65, 69, ...
# Tugs are 50s: 50, 51, 59, ...

# I am going to use Logistic Regression to predict the class of the ship based on the audio data.

# Extract features
print("Extracting features...")
def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed

X = [extract_features(entry['audio'], entry['sample_rate']) for entry in dataset]
print("Features extracted.")
# Flatten as Logistic Regression cannot handle 2D features.

# Use class_id as target
# Function to group classes
def group_classes(class_id):
    return class_id // 10 * 10

y = [group_classes(entry['class_id']) for entry in dataset]
print("Classes grouped.")
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split.")
# Data pre-processing and model pipeline
print("Training model...")
model = Pipeline([
    ("scaler", StandardScaler()),  # Normalize the MFCC features.
    ("logreg", LogisticRegression())
])

# Train the model
model.fit(np.array(X_train), np.array(y_train))

# Evaluate the model
print(classification_report(y_test, model.predict(X_test)))