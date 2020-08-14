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
		self.node = 10 # The number of nodes in the output layer
		self.track = [] # Keeps track of layer order, i.e Conv./Pooling/FC
		self.regLossParam = 1e-3 # Regularization strength
		self.learning_rate = 0.1
		self.fc_weights = []
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
		self.track.append('c')

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
		self.lengths.append(prevl/r)
		self.widths.append(prevw/r)
		self.depths.append(prevd)
		self.track.append('p')

	def FCLayer(self, n_nodes):
		"""
		Creates a fully connected layer
		n - the no.of nodes in the output layer.
		input_fc - the input to the fully connected layer.
		"""
		# Depth, width and length of previous layer
		prevd = int(self.depths[-1])
		prevw = int(self.widths[-1])
		prevl = int(self.lengths[-1])

		# flatten the input
		input_fc = np.zeros((prevd, prevw, prevl))
		input_fc = input_fc.flatten()
		len_input_fc = len(input_fc)

		# Initialise the weights and biases for the FC layer
		self.fc_weights = np.random.randn(len_input_fc, n_nodes)
		#self.fc_bias = np.zeros(n_nodes)

		# Store them
		self.weights.append(self.fc_weights)
		self.strides.append(0)
		self.recfield.append(0)
		self.lengths.append(1)
		self.widths.append(len_input_fc)
		self.depths.append(1)
		self.track.append('f')
		self.node = n_nodes

	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		return np.maximum(0, inputArray)

	def dataLoss(self, predResults, trueResults):
		"""
		Returns the data loss. Cross-Entropy loss function (Softmax Classifier).
		"""
		# L2 loss
		loss = 0
		sum = 0
		for i in range(len(predResults)):
			sum += math.exp(predResults[i])
		correct = np.argmax(trueResults)
		loss = (-1)*(math.log((math.exp(predResults[correct]))/sum))
		return loss

	def ConvOutput(self, prevOut, W, s, r, d, w, l):
		"""
		Returns the output of the Convolutional Layer.
		prevOut - Output from the previous layer
		W = Weight of this layer
		"""
		prevOut = np.array(prevOut)
		d = int(d)
		w = int(w)
		l = int(l)
		volOutput = np.zeros((d, w, l))
		if(len(W.shape)<4):
			f = 1
		else:
			f = W.shape[0]
		
		for i in range(f):   #Run loop to create f-filters
			for k in range(w):  #Convolve around width
				for m in range(l):   #Convolve around length
					#for j in range(d):   #Run over entire depth of prevOut volume
					volOutput[i][k][m] += np.sum(np.multiply(W[i][:][:][:], prevOut[:, k*s: k*s + r, m*s: m*s + r])[:,:,:])

		volOutput = np.array(volOutput)
		return volOutput

	def PoolOutput(self, prevOut, W, s, r, d, w, l):
		"""
		Returns the output of the Pooling Layer.
		prevOut - Output from the previous layer
		W = Weight of this layer, since there is no Weight matrix for MaxPooling, it is None
		"""
		prevOut = np.array(prevOut)
		d = int(d)
		w = int(w)
		l = int(l)
		volOutput = np.zeros((d, w, l))
		for j in range(d):
			for k in range(w):
				for m in range(l):
					volOutput[j][k][m] = np.amax(prevOut[j, k*r: (k + 1)*r, m*r: (m + 1)*r])

		volOutput = np.array(volOutput)
		return volOutput

	def FCOutput(self, prevOut, W, s, r, d, w, l):
		"""
		Creates a fully connected layer; implements forward pass and
		backpropagation.
		n_nodes - the no.of nodes in the fully connected layer.
		input_fc - the input to the fully connected layer.
		"""
		# flatten the input
		prevOut = prevOut.flatten()
		#len_input_fc = len(prevOut)

		totals = np.dot(prevOut, W) #+ self.fc_bias

		# Output from the FC layer
		self.output_fc = self.activFunc(totals)
		return self.output_fc

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
			d = self.depths[i+1]
			w = self.widths[i+1]
			l = self.lengths[i+1]
			if (self.track[i] == 'c'):
				h = self.activFunc(self.ConvOutput(h, W, s, r, d, w, l))
			elif (self.track[i] == 'p'):
				h = self.PoolOutput(h, W, s, r, d, w, l)
			else:
				h = self.FCOutput(h, W, s, r, d, w, l)

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
			return self.FCOutput(h, W, s, r, d, w, l)

	def FCGD(self, index, trueResults): #FC layer Gradient Descent
		input_fc = self.getVolumeOutput(index - 1)

		# store the shape of input to be used for backpropagation
		input_fc_shape = input_fc.shape

		# flatten the input
		input_fc = input_fc.flatten()

		W = self.weights[index - 1]
		totals = np.dot(input_fc, W)

		h = 0.001 * np.ones(self.output_fc.shape)
		d_L_d_out = (self.dataLoss(self.output_fc + h, trueResults) - self.dataLoss(self.output_fc - h, trueResults))/(2*h)
		d_out_d_t = np.where(totals <= 0, 0, 1)
		d_t_d_w = input_fc
		d_t_d_inputs = W
		d_L_d_t = d_L_d_out * d_out_d_t

		# Gradients of loss wrt Weights of FC layer and Input of FC layers
		# d_L_d_t.shape = (n_nodes,1)
		# Adding appropriate axes to d_L_d_t and d_t_d_w(same as input_fc) for . product
		d_L_d_w = np.dot(d_t_d_w[np.newaxis].T, d_L_d_t[np.newaxis])

		# d_L_d_inputs should have the dimensions of input_fc
		d_L_d_inputs = np.dot(d_t_d_inputs, d_L_d_t)

		# The dimension of d_L_d_inputs is (len_input_fc,), so, changing the shape so it can be given to maxpool's backprop.
		d_L_d_inputs_final = d_L_d_inputs.reshape(input_fc_shape)

		W -= self.learning_rate * d_L_d_w
		self.weights[index - 1] = W
		return d_L_d_inputs_final

	def PoolGD(self, dLdOut, index):
		"""
		Function that backpropagates gradients through the MaxPooling layer
		dLdOut is the differential of Loss wrt the Output where Output here refers to the output of the MaxPooling layer
		This function thus finds dLdI which is the differential of Loss wrt the Input where Input here refers to input to MaxPool layer.
		"""
		input_vol = self.getVolumeOutput(index - 1)
		s = self.strides[index - 1]
		r = self.recfield[index - 1]
		d = dLdOut.shape[0]
		w = dLdOut.shape[1]
		l = dLdOut.shape[2]

		#Convert the numbers to int, as the for loops below will report errors if this is not done 
		d = int(d)
		w = int(w)
		l = int(l)

		#Keep track of the depth and spatial indices of where the maximum element is, in the sub arrays taken for pooling
		d_ind = []
		spatial_ind = []
		#Keep track of which sub array is being taken for max pooling
		track_w = []
		track_l = []

		dLdI = np.zeros((int(self.depths[index - 1]), int(self.lengths[index - 1]), int(self.widths[index - 1])))
		replace = dLdOut.flatten()
		for j in range(d):
			for k in range(w):
				for m in range(l):
					spatial_ind.append(np.where(input_vol[j, k*r: (k + 1)*r, m*r: (m + 1)*r] == input_vol[j, k*r: (k + 1)*r, m*r: (m + 1)*r].max()))
					track_l.append(m)
					track_w.append(k)
					d_ind.append(j)

		#Initialise correct values in dLdI array
		for i in range(len(replace)):
			width = spatial_ind[i][0][0]    # Note the (width) spatial index of the maximum element of the sub array 
			width += track_w[i]*r  # Add the (width) location depending on which sub array was taken for max pooling
			length = spatial_ind[i][1][0]   # Note the (length) spatial index of the maximum element of the sub array 
			length += track_l[i]*r  # Add the (length) location depending on which sub array was taken for max pooling
			depth = d_ind[i]  # Note the depth index of the maximum element of the sub array 
			dLdI[depth][width][length] = replace[i]

		return dLdI
	
	# Helper functions for convBackProp()
	def convolve(self, inputLayer, convFilter):
		"""
		Returns the convoluted output convFilter on inputLayer.
		Both are two dimensional matrices square matrices.
		inputLayer - (n, n)
		convFilter - (f, f)
		"""
		# Dimensions of the input matrices
		n = inputLayer.shape[0]
		f = convFilter.shape[0]
		
		# Defining the shape of the output matrix
        l = (n-f) + 1
        output_matrix = np.zeros((l, l))
        s = 1

        # Convolving
        for row in range(l):
            for col in range(l):
                output_matrix[row][col] = np.sum(np.multiply(inputLayer[row:row+f,col:col+f], convFilter))

		return output_matrix
		
	def fullConvolve(self, inputLayer, convFilter):
		"""
		Returns the full convoluted output of convFilter on inputLayer.
		"""
		
		# Dimensions of the input matrices
		n = inputLayer.shape[0]
		f = convFilter.shape[0]
		
		# Creating padding for the inputLayer matrix
		padding = f - 1
		new_dim = n + 2*padding
		
		padded_input = np.zeros([new_dim, new_dim])
		padded_input[padding:new_dim - padding, padding:new_dim - padding] = inputLayer
		
		# Now convolve padded_input with convFilter
		output_matrix = self.convolve(padded_input, convFilter)
		
		return output_matrix
	
	def rotate180(self, matrix):
		"""
		Rotates matrix by 180 degrees in the plane.
		Takes only two dimensional matrices.
		"""
		return np.rot90(matrix, 2)
		
	def ConvGD(self, dLdoutput, index):
		"""
		Function that backpropagates through a convolutional layer.
		index = index of the current layer
		dLdoutput = Gradient of the loss function wrt the output of the current layer (channel, row, col)
		Returns dLdinput.
		"""
		X = self.getVolumeOutput(index-1) # Input to the current layer (channel, row, col)
		W = self.weights[index - 1] # Weights of the current layer (numFilter, channel, row, col)

		dLdX = np.empty(X.shape)
		dLdW = np.empty(W.shape)
  
		dLdout = np.copy(dLdoutput)
		dLdout[dLdout < 0] = 0
		
		# Loop over the filters
		numFilters = W.shape[0]
		
		for fil_ter in range(numFilters):
			filter_output = dLdout[fil_ter]
			
			# Loop over the channels
			for channel in range(W.shape[1]):
				filter_layer = W[fil_ter][channel]
				dWLayer = self.convolve(X[channel], filter_output)
				dXLayer = self.rotate180(self.fullConvolve(self.rotate180(filter_layer), filter_output))
				
				# Combine these and return in arrays
				dLdW[fil_ter][channel] = dWLayer
				dLdX[channel] = dXLayer
		
		W -= self.learning_rate * dLdW
		self.weights[index - 1] = W				
		return dLdX


	def backPropagation(self, input, trueResults):
		"""
		Updates weights by carrying out backpropagation.
		trueResults = the expected output from the neural network.
		"""
		for i in range(len(input)):
			self.inputImg = np.array(input[i])
			if(len(self.inputImg.shape)<3):
				a = self.inputImg
				self.inputImg = a.reshape(1, a.shape[0], a.shape[1])
			# Called once so that all weights are initialised, just in case if not done before
			out = self.getVolumeOutput(len(self.weights))
			nPrev = len(self.weights) # Index keeping track of the previous layer
			doutput = self.FCGD(nPrev, trueResults[i])
			nPrev -= 1
			
			# Loop over the layers
			while nPrev - 1 >= 0:
				if(self.track[nPrev - 1] == 'p'):
					dhidden = self.PoolGD(doutput, nPrev)
				else:
					dhidden = self.ConvGD(doutput, nPrev)
				doutput = dhidden # Move to the previous layer
				nPrev -= 1  			
	
	def train(self, input, Y, epochs):
		"""
		Train the neural network.
		Y = the expected results from the neural network.
		epochs = the number of times the neural network should 'learn'.
		"""
		# Run backPropagation() 'epochs' number of times.
		for i in range(epochs):
			self.backPropagation(input, Y)
			print('Epoch Number: ',i + 1,' done.')
		print("Training Complete.")

	def accuracy(self, X, Y):
		"""
		Function that takes in test data and results and calculates the accuracy of the Network.
		"""
		y = []
		cor = 0
		correct = 0
		for i in range(len(X)):
			self.inputImg = np.array(X[i])
			if(len(self.inputImg.shape)<3):
				a = self.inputImg
				self.inputImg = a.reshape(1, a.shape[0], a.shape[1])
				y.append(self.getVolumeOutput(len(self.weights)))
		Y = np.array(Y)
		y = np.array(y)
		if (np.max(y)==0):
			y /= 1.0
		else:
			y /= np.max(y)
		for i in range(len(Y)):
			correct = np.argmax(Y[i])
			if (np.argmax(y[i]) == correct):
				cor += 1
		cor /= len(Y)
		print('Accuracy = ',cor*100)