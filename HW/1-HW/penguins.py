from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.image as mpimg
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout


penguin_data = pd.read_csv('penguins_size.csv')

# Drop the rows with missing values
penguin_data = penguin_data.dropna()  
penguin_data.isna().sum() # Check for missing values


# preprocessing the data
# Create separate LabelEncoders
label_encoder_species = LabelEncoder()
label_encoder_island = LabelEncoder()
label_encoder_sex = LabelEncoder()

# Fit and transform for each categorical column
penguin_data['species'] = label_encoder_species.fit_transform(penguin_data['species'])
penguin_data['island'] = label_encoder_island.fit_transform(penguin_data['island'])
penguin_data['sex'] = label_encoder_sex.fit_transform(penguin_data['sex'])


# Split the data into training and testing sets
# Assuming 'sex' is the target feature and the model is a binary classification
X = penguin_data.drop(columns=['sex']).values # Features: everything except 'sex'
Y = penguin_data['sex'].values # Target: 'sex'
print(np.unique(Y))
X_training, X_value, Y_training, Y_value = train_test_split(X, Y, test_size=0.1)


# using the same network as before from the diabetes
# Training the model 


neural_network = Sequential()
neural_network.add(Dense(32, input_dim=X_training.shape[1], activation='relu'))
neural_network.add(Dropout(0.5)) # 50% dropout rate
neural_network.add(Dense(24, activation='relu'))
neural_network.add(Dropout(0.5)) # 50% dropout rate
neural_network.add(Dense(8, activation='relu'))
neural_network.add(Dense(1, activation='sigmoid'))
neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Creating the optimizer with a custom learning rate
optimizer = Adam(learning_rate=0.001)

# Compiling the model
neural_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fitting data with smaller batch size
history = neural_network.fit(X_training, Y_training, batch_size=8, validation_data=(X_value, Y_value), epochs=500)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')


