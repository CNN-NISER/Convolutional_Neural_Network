import numpy as np
from convnet import ConvNet

# Create instance of NeuralNetwork
model = ConvNet()
x = np.random.randn(3,3,3)
model.addInput(x) # Input layer
model.cvolume(1,2,2) # Add Convolutional volume
model.pmaxvolume(2) # Add Pooling layer
model.FCLayer(3)
print(model.getVolumeOutput(3))  # Get final output layer
