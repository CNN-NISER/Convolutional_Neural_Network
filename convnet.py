import numpy as np
import math

# Ensuring reproducibility
np.random.seed(0)

class ConvNet():
	
	def __init__(self):
		self.inputImg = None # The input data
		self.strides = [] # The stride length of each layer for convolution
		self.recfield = [] # The receptive field in each layer for convolution
		self.lengths = []
		self.widths = []
		self.depths = []
		self.weights = [] # The weights for convolution filters
		self.nodes = [] # The number of nodes in each layer of the FC
		self.fcweights = [] # The weights and biases (as tuples) for the fc layer
		self.regLossParam = 1e-3 # Regularization strength
		
	def addInput(self, inpImage):  # Assign the input image
		inpImage = np.array(inpImage)
		self.inputImg = inpImage
		if(len(inpImage.shape)<3):
			num3 = 1
			numrows = inpImage.shape[0]
			numcols = inpImage.shape[1]
		else:
			num3 = inpImage.shape[0]
			numrows = inpImage.shape[1]
			numcols = inpImage.shape[2]

		self.lengths.append(numcols)
		self.widths.append(numrows)
		self.depths.append(num3)
		
	def volume(self, s, r, f):
		"""
		Creates a new Conv. volume.
		s - stride length for convolving the previous layer to create this new volume
		r - receptive field for convolving the previous layer to create this new volume
		f - number of filters, or in other words, the depth of this new volume
		"""
		
		# Depth of previous layer
		prevd = self.depths[-1]
		prevw = self.widths[-1]
		prevl = self.lengths[-1]
		
		# Initializing the weights
		W = []
		#b = np.zeros((1, stre))
		for i in range(f):
			W.append(np.random.randn(prevd, r, r))
		
		W = np.array(W)
		numrows = (prevw - r)/s + 1
		numcols = (prevl - r)/s +1
		num3 = f
		
		# Store them
		self.weights.append(W)
		self.strides.append(s)
		self.recfield.append(r)
		self.lengths.append(numcols)
		self.widths.append(numrows)
		self.depths.append(num3)


	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		return np.maximum(0, inputArray)


	def hiddenVolumeOutput(self, prevOut, W, s, r):
		"""
		Returns the output of Convolutional Layers.
		prevOut - Output from the previous layer
		W, b = Weight and bias of this layer
		"""
		prevOut = np.array(prevOut)

		if(len(W.shape)<4):
			f = 1
		else:
			f = W.shape[0]

		if(len(prevOut.shape)<3):
			d = 1
			w = prevOut.shape[0]
			l = prevOut.shape[1]
		else:
			d = prevOut.shape[0]
			w = prevOut.shape[1]
			l = prevOut.shape[2]
		
		wid = int((w - r)/s + 1)
		leng = int((l - r)/s + 1)

		volOutput = np.zeros((f, leng, wid))
		for i in range(f):   #Run loop to create f-filters
			for k in range(int((w - r)/s + 1)):  #Convolve around width
				for m in range(int((l - r)/s + 1)):   #Convolve around length
					for j in range(d):   #Run over entire depth of prevOut volume
						volOutput[i][m][k] += np.sum(np.multiply(W[i][:][:][:], prevOut[:, k*s: k*s + s + 1, m*s: m*s + s + 1])[:,:,:])
		volOutput = np.array(volOutput)

		return self.activFunc(volOutput)
		
	def finalVolumeOutput(self, prevOut, W, s, r):
		"""
		Returns the output of the final layer.
		Similar to hiddenLayerOutput(), but without 
		the activation function.
		"""
		prevOut = np.array(prevOut)

		if(len(W.shape)<4):
			f = 1
		else:
			f = W.shape[0]

		if(len(prevOut.shape)<3):
			d = 1
			w = prevOut.shape[0]
			l = prevOut.shape[1]
		else:
			d = prevOut.shape[0]
			w = prevOut.shape[1]
			l = prevOut.shape[2]

		x = int((w - r)/s + 1)
		y = int((l - r)/s + 1)
		print(f,y,x)

		finalVolOutput = np.zeros((f, y, x))  #d,w,l
		for i in range(f):   #Run loop to create f-filters
			for k in range(int((w - r)/s + 1)):  #Convolve around width
				for m in range(int((l - r)/s + 1)):   #Convolve around length
					for j in range(d):   #Run over entire depth of prevOut volume
						finalVolOutput[i][m][k] += np.sum(np.multiply(W[i][:][:][:], prevOut[:, k*s: k*s + s + 1, m*s: m*s + s + 1])[:,:,:])
		finalVolOutput = np.array(finalVolOutput)
		#layerOutput = np.dot(prevOut, W) #==================
		return finalVolOutput


	def getVolumeOutput(self, n):
		"""
		Returns the output of the nth volume of the ConvNet.
		"""
		penLayer = len(self.weights) - 1 # The penultimate volume
		
		# h stores the output of the current layer
		h = np.array(self.inputImg)

		# Loop through the hidden layers
		for i in range(min(n, penLayer)):
			W = self.weights[i]   #======================
			s = self.strides[i]
			r = self.recfield[i]
			h = self.hiddenVolumeOutput(h, W, s, r) #===================

		# Return the output
		if n <= penLayer:
			return h
		else:
			W = self.weights[n-1]
			s = self.strides[n-1]
			r = self.recfield[n-1]
			return self.finalVolumeOutput(h, W, s, r) #==================



#===============================================================================================================================================
#===============================================================================================================================================