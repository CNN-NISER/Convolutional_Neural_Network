import numpy as np
from convnet import ConvNet

# Create instance of NeuralNetwork
model = ConvNet()
x = np.random.randn(3,3,3)
model.addInput(x) # Input layer
model.volume(1,2,1) # Add Convolutional volume
model.getVolumeOutput(1)  # Get final output from convolutional layer
