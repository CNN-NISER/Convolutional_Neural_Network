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
		#self.fcweights = [] # The weights and biases (as tuples) for the fc layer
		self.regLossParam = 1e-3 # Regularization strength

			# Added
			self.fc_weights = []
			#self.fc_bias = []
			self.fc_output = []

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

	def cvolume(self, s, r, f):
		"""
		Creates a new Conv. volume.
		s - stride length for convolving the previous layer to create this new volume
		r - receptive field for convolving the previous layer to create this new volume
		f - number of filters, or in other words, the depth of this new volume
		"""
		# Depth, width and length of previous layer
		prevd = self.depths[-1]
		prevw = self.widths[-1]
		prevl = self.lengths[-1]

		# Initializing the weights
		W = []
		#b = np.zeros((1, stre))
		for i in range(f):
			W.append(np.random.randn(prevd, r, r))

		W = np.array(W)

		# The dimensions of the layer after convolution with the above weight array
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

	def pmaxvolume(self, r):
		"""
		Creates a new max pooling layer.
		r - the receptive field around which the max value has to be taken.
		    E.g - If r = 2, max pooling is done withing 2x2 sub matrices.
		"""
		# Depth, width and length of previous layer
		prevd = self.depths[-1]
		prevw = self.widths[-1]
		prevl = self.lengths[-1]

		# Store them
		self.weights.append(None)
		self.strides.append(r)
		self.recfield.append(r)

		#The dimensions of the layer after pooling
		self.lengths.append(prevl/r)
		self.widths.append(prevw/r)
		self.depths.append(prevd)

	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		return np.maximum(0, inputArray)

	def dataLoss(self, predResults, trueResults):
		"""
		Returns the data loss.
		"""
		# L2 loss
		loss = np.square(trueResults - predResults)
		return loss/len(trueResults)


	def hiddenVolumeOutput(self, prevOut, W, s, r, d, w, l):
		"""
		Returns the output of the Convolutional/Pooling Layer.
		prevOut - Output from the previous layer
		W = Weight of this layer
		"""
		prevOut = np.array(prevOut)
		d = int(d)
		w = int(w)
		l = int(l)
		volOutput = np.zeros((d, l, w))

		if(W is None):
			for j in range(d):   #Run over entire depth of prevOut volume
				for k in range(w):  #Convolve around width
					for m in range(l):   #Convolve around length
						volOutput[j][m][k] = np.amax(prevOut[j, k*r: (k + 1)*r, m*r: (m + 1)*r])

			volOutput = np.array(volOutput)
			return volOutput

		else:
			if(len(W.shape)<4):
				f = 1
			else:
				f = W.shape[0]

			for i in range(f):   #Run loop to create f-filters
				for k in range(w):  #Convolve around width
					for m in range(l):   #Convolve around length
						#for j in range(d):   #Run over entire depth of prevOut volume
						volOutput[i][m][k] += np.sum(np.multiply(W[i][:][:][:], prevOut[:, k*s: k*s + s + 1, m*s: m*s + s + 1])[:,:,:])

			volOutput = np.array(volOutput)
			return self.activFunc(volOutput)

	def finalVolumeOutput(self, prevOut, W, s, r, d, w, l):
		"""
		Returns the output of the final layer.
		Similar to hiddenLayerOutput(), but without the activation function.
		"""
		prevOut = np.array(prevOut)
		d = int(d)
		w = int(w)
		l = int(l)
		finalVolOutput = np.zeros((d, l, w))  #d,w,l
		print(d,w,l)
		if(W is None):
			for j in range(d):   #Run over entire depth of prevOut volume
				for k in range(w):  #Convolve around width
					for m in range(l):   #Convolve around length
						finalVolOutput[j][m][k] = np.amax(prevOut[j, k*r: (k + 1)*r, m*r: (m + 1)*r])

			finalVolOutput = np.array(finalVolOutput)
			return finalVolOutput

		else:
			if(len(W.shape)<4):
				f = 1
			else:
				f = W.shape[0]

			for i in range(f):   #Run loop to create f-filters
				for k in range(w):  #Convolve around width
					for m in range(l):   #Convolve around length
						#for j in range(d):   #Run over entire depth of prevOut volume
						finalVolOutput[i][m][k] += np.sum(np.multiply(W[i][:][:][:], prevOut[:, k*s: k*s + s + 1, m*s: m*s + s + 1])[:,:,:])

			finalVolOutput = np.array(finalVolOutput)
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
			W = self.weights[i]
			s = self.strides[i]
			r = self.recfield[i]
			d = self.depths[i]
			w = self.widths[i]
			l = self.lengths[i]
			h = self.hiddenVolumeOutput(h, W, s, r, d, w, l)

		# Return the output
		if n <= penLayer:
			return h
		else:
			W = self.weights[n-1]
			s = self.strides[n-1]
			r = self.recfield[n-1]
			d = self.depths[n]
			w = self.widths[n]
			l = self.lengths[n]
			return self.finalVolumeOutput(h, W, s, r, d, w, l)

	def fullyConn(self, n_nodes, input_fc, trueResults, learning_rate):

		"""
		Creates a fully connected layer; implements forward pass and
		backpropagation.
		n_nodes - the no.of nodes in the fully connected layer.
		input_fc - the input to the fully connected layer.
		"""
		# store the shape of input to be used for backpropagation
		input_fc_shape = input_fc.shape

		# flatten the input
		input_fc = input_fc.flatten()
		len_input_fc = len(input_fc)

		# Initialise the weights and biases for the FC layer
		self.fc_weights = np.random.randn(len_input_fc, n_nodes)
		#self.fc_bias = np.zeros(n_nodes)

		totals = np.dot(input_fc, self.fc_weights) #+ self.fc_bias

		# Output from the FC layer
		self.output_fc = activFunc(totals)

		h = 0.001 * np.ones(self.output_fc.shape)
		d_L_d_out = (self.dataLoss(self.output_fc + h, trueResults) - self.dataLoss(self.output_fc - h, trueResults))/(2*h)
		d_out_d_t = np.where(totals <= 0, 0, 1)
		d_t_d_w = input_fc
		d_t_d_inputs = self.fc_weights
		d_L_d_t = d_L_d_out * d_out_d_t

		# Gradients of loss wrt Weights of FC layer and Input of FC layers
		# The shape of input_fc is now (len_input_fc, 1) after flattening
		# d_L_d_t.shape = (n_nodes,1)
		# Adding appropriate axes to d_L_d_t and d_t_d_w(same as input_fc) for . product
		d_L_d_w = np.dot(d_t_d_w[np.newaxis].T, d_L_d_t[np.newaxis])

		# d_L_d_inputs should have the dimensions of input_fc
		d_L_d_inputs = np.dot(d_t_d_inputs, d_L_d_t)

		# The dimension of d_L_d_inputs is (len_input_fc,), so, changing the shape so it can be given to maxpool's backprop.
		d_L_d_inputs_final = d_L_d_inputs.reshape(input_fc_shape)

		self.fc_weights -= learning_rate * d_L_d_w

		return d_L_d_inputs_final
