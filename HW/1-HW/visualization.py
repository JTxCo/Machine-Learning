from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
# Your existing code
dataset = loadtxt('diabetes.csv', delimiter=',', skiprows=1)
X_training = dataset[10:, 0:8]
y_training = dataset[10:, 8]
neural_network = Sequential()
neural_network.add(Dense(12, input_shape=(8,), activation='relu'))
neural_network.add(Dense(8, activation='relu'))
neural_network.add(Dense(1, activation='sigmoid'))
neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = neural_network.fit(X_training, y_training, epochs=150, batch_size=10)

# Visualization
plot_model(neural_network, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

img = mpimg.imread('model_plot.png')
plt.figure(figsize=(10,10))
imgplot = plt.imshow(img)
plt.show()
