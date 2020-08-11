import numpy as np
from convnet import ConvNet
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
# For skeptics, we use tensorflow only for importing the MNIST data easily. The Conv Net architecture and calculations are done with only NumPy

(x, y), (x2, y2) = datasets.mnist.load_data()
# Normalise data
x = x/255 
x2 = x2/255

# Create instance of NeuralNetwork
model = ConvNet()
X = np.random.randn(1,28,28)
model.addInput(X) # Input layer
model.cvolume(1,3,5) # Add Convolutional volume (stride length, receptive field, filters)
model.pmaxvolume(2) # Add Pooling layer (receptive field)
model.FCLayer(10)  # Add FC Layer (number of classifiers)
print(model.getVolumeOutput(3))  # Get final output layer
# Represent y in the required result format
results = np.zeros((len(y),10))  
for i in range(len(y)):
    results[i,y[i]] = 1

model.train(x[0:100], results[0:100], 10) # Train model with 100 images for 10 epochs
model.accuracy(x[100:110], results[100:110])  # Predict accuracy using test data
