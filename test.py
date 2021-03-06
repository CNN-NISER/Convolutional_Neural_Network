import numpy as np
from convnet import ConvNet
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split

# (x,y) - Training data of 60000 images, (x2, y2) - Testing data od 10000 images
(x, y), (x2, y2) = datasets.mnist.load_data()
# Normalise data
x = ((x/255) - 0.5)
x2 = ((x2/255) - 0.5)

# Create instance of NeuralNetwork
model = ConvNet()
X = np.random.randn(1,28,28)
model.addInput(X) # Input layer
model.cvolume(1,3,10) # Add Convolutional volume (stride length, receptive field, filters)
model.pmaxvolume(2) # Add Pooling layer (receptive field)
model.FCLayer(10)  # Add FC Layer (number of classifiers)
# Get final output layer. It is advised to run it once before training, so that all variables are initialised.
print('Test Run Output: ',model.getVolumeOutput(3))

#Since we test the CNN with MNIST data, we write the target output in the required format before sent to training/testing.
results = np.zeros((len(y),10))  
for i in range(len(y)):
    results[i,y[i]] = 1

model.train(x[0:2000], results[0:2000], 5) # Train model with 2000 images for 5 epochs
model.accuracy(x[2000:4000], results[2000:4000])  # Predict accuracy using test data
