import numpy as np
from convnet import ConvNet
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

# Create instance of NeuralNetwork
model = ConvNet()
x = np.random.randn(3,3,3)
model.addInput(x) # Input layer
model.cvolume(1,2,2) # Add Convolutional volume
model.pmaxvolume(2) # Add Pooling layer
model.FCLayer(10)
print(model.getVolumeOutput(3))  # Get final output layer