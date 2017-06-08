import numpy as np
import random as rand
import time
from Neuron import *
path = '/home/jsurg/Programs/python/NeuralNet/resource/'

def sig(x, deriv=False):
	if(deriv):
		return x*(1-x)
	return 1/(1 + np.exp(-x))
	
class Net(object):
	""" 
	Neural Network Implementation
	
		Note: The weights can be created and accessed in various ways. A connection
		object can store the weights (coming soon!), they can be stored in matrices in the usual way,
		or we can directly access weights initiated and stored in a neuron object.
		The weight stored in a neuron exists in the first hidden layer to the last layer (output).
		So, when calculating with these weights, they must be used to multiply values that
		are INCOMING to the neuron. Thus, the weights for connections from some layer
		A to a layer B are stored in the neurons in the B layer. Calculate accordingly,
		looping over each layer and using the next layer to access the weights.
	
	"""
	def __init__(self, topology, inputs, targs, matrix=False, rate=0.5, load_=False, weightFile=None, test=False):
		self.num_inputs = topology[0]
		self.num_outputs = topology[-1]
		self.using_matrix = matrix
		assert(len(inputs) == self.num_inputs)
		self.nextLayer = None
		self.prevLayer = None
		self.thisLayer = None
		self.max_layer = len(topology)
		self.input_list = [inputs]
		self.matrix_inputs = np.array([self.input_list]).T
		self.target_values = targs
		self.num_weight_layers = len(topology) - 1
		self.matrix_outputs = []
		self.layerNumber = 0
		self.matrix_weights = [[] for x in xrange(self.max_layer - 1)] 
		self.eta = rate
		self.vector_grad = [[] for x in xrange(self.max_layer)]
		self.testing = test
		self.Error = 0
		self.thresh = 0.0000009
		self.trained = False
		self.vectors = [[] for x in range(len(topology))] # one list of values for each layer
		self.load = load_
		self.weight_file_name = weightFile
		self.weight_file = None
		
		#initialize the list of lists with the number of sub-lists == length of topology
		self.m_layers = [[] for i in xrange(len(topology))]
		
		#attach input Neurons...
		for x in range(self.num_inputs):
			self.m_layers[0].append(Neuron(layerNumber=0, _inputs=1, val=inputs[x], firstLayer=True, testing=test))
			self.vectors[0].append(inputs[x])
			if(self.testing):
				print("attached input Neuron in first layer")
			
		#attach hidden Neurons using topology....Create a new connection
		for layer in range(1, len(topology) - 1):
			for n in range(topology[layer]):
				#need one new connection per node.			
				if layer + 1 < len(topology): #not last hidden layer
					self.m_layers[layer].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False, testing=test))
					self.layerNumber = layer
					if(self.testing):
						print(("attached hidden Neuron in layer number %d") % (layer + 1))
				else: #last hidden layer
					self.m_layers[-2].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False, testing=test))
					if(self.testing):
						print(("attached hidden Neuron in layer number %d") % (layer + 1))
					self.layerNumber = layer
		
		#attach output Neurons...
		for x in range(self.num_outputs):
			self.m_layers[-1].append(Neuron(layerNumber=self.max_layer, _inputs=topology[-2], firstLayer=False, testing=test))
			self.layerNumber = self.max_layer
			if(self.testing):
				print("attached output Neuron in last layer")
			
		#Set up matrix weights if using matrices		
		if(matrix and load_ == False):
			np.random.seed(int(time.time()))
			# setting up the matrix holding all weights, with x being the (layer - 1)st connection.
			for x in range(1, len(topology)):
				self.matrix_weights[x-1] = np.random.random((topology[x], topology[x-1]))
				
			if(self.testing):
				print("-"*20 + "MATRIX WEIGHTS" + "-"*20)
				print(self.matrix_weights)
			
			#create input matrix
			self.matrix_inputs = np.array(self.input_list).T
			
			if(self.testing):
				print("-"*20 + "INPUT MATRIX" + "-"*20)
				print(self.matrix_inputs)		
			
		elif(matrix and load_):
			#load the weights from file
			assert(type(self.weight_file_name) is str)
			self.load_weights()
			print "Loaded Matrix Weights successfully!"		
						
	
	def setMatrixInputs(self, matrix):
		self.matrix_inputs = matrix
		
		
	def setTargetValues(self, targs):
		assert(len(targs) == self.num_outputs)
		self.target_values = targs


	def printNodes(self):
		for layer in self.m_layers:
			for neuron in layer:
				print neuron
				
	
	def feedForward(self):
		if(self.using_matrix):
			self.ffMatrix()
		else:
			self.feedForward_Normal()
	
	
	def feedForward_Normal(self):
		#get vals from prevLayer, multiply EACH by weights in this layer and sum.
		nextVal, sum, this_node, this_layer = 0, 0, 0, 0
		w = []
		self.nextLayer = self.m_layers[1]
		self.prevLayer = self.m_layers[0]
			
		for layer in self.m_layers:
			if this_layer == self.max_layer:
				break
			prevLayer = self.m_layers[this_layer - 1]
			this_layer += 1
			w = []
			for neuron in layer:
				#skip the first layer.
				if neuron.getLayerNumber() == 0:
					break
				w = neuron.getWeights()
				values = [n.getValue() for n in prevLayer] # Values in previous layer to list 
				
				# Loop over values and multiply by weights in next layer.
				for i in range(len(values)):				
					sum += values[i]*w[i]
				valp = neuron.processFunction(sum, function="sigmoid", deriv=False)
				neuron.setValue(valp)
				sum = 0
				valp = 0
				w = []
				
				values = []
		
		if(self.testing):
			self.print_outputs()	
	
	def ffMatrix(self):
		index, wLayer = 1, 0,
		layer = 0
		self.setLayers(layer)
		num_Hlayers = self.max_layer - 1
				
		#calculate all hidden layer values (and outputs)
		for layer in range(0, self.max_layer - 1):
			index = 0
			self.setLayers(layer) #layer started at 0
			weights = np.array(self.matrix_weights[layer])
			if(self.testing):
				print "weights = {}".format(weights)
				print "vectors[{}] = {}".format(layer, self.vectors[layer])
			
			next_matrix = sig(weights.dot(self.vectors[layer]))
			
			if(self.testing):
				print "Next matrix is: {}".format(next_matrix)
			
			# Storing the vector gotten by calc above in vectors
			self.vectors[layer + 1] = next_matrix
			
			#assert that there is a 1 to 1 correspondence between vectors and number of neurons in next layer.
			assert(len(self.vectors[layer + 1]) == len(self.nextLayer))
			index = 0
			#set values in neurons themselves
			for value in self.vectors[layer + 1]:
				self.nextLayer[index].setValue(value)
				index += 1
		
		
	def setLayers(self, layernum):
		self.nextLayer = self.m_layers[layernum + 1]
		self.prevLayer = self.m_layers[layernum - 1]
		self.thisLayer = self.m_layers[layernum]
		
		
	def feedForward_Matrix(self):
		#set up the layers
		ind, layer, this_layer, this_node = 1, 0, 0, 0
		self.nextLayer = self.m_layers[layer + 1]
		self.prevLayer = self.m_layers[layer - 1]
				
		for x in xrange(0, self.num_weight_layers ):
			next_matrix = []
			inp = np.array(self.matrix_inputs)
			if(self.testing):
				print "inp/matrix_inputs = {}".format(inp)
				print "weight_matrix = {}".format(self.matrix_weights[x])
			weight_matrix = np.array(self.matrix_weights[x])
			
			if(inp.shape[1] == weight_matrix.shape[0]):
				next_matrix = sig(np.dot(weight_matrix, inp))
			else:
				next_matrix = next_matrix * sig(float(inp), deriv=False)
			print "next_matrix = {}".format(next_matrix)
			self.setMatrixInputs(next_matrix)
			self.vectors[ind] = next_matrix
			ind += 1
			
			#skip over the input layer.
			layer += 1
			assert(len(next_matrix) == len(self.m_layers[layer]))
			ind = 0	
			#set values in neurons.
			i = 0
			for neuron in self.m_layers[layer]:
				neuron.setValue(next_matrix[i])
				i += 1
			
		#set outputs
		self.matrix_outputs = next_matrix
		#i = 0
		#for n in self.m_layers[-1]:
			#n.setValue(self.matrix_outputs[i])
			#i += 1
		if(self.testing):
			vals = [i.getValue() for i in self.m_layers[-1]]
			print "Forward pass outputs: {}".format(vals)
		
						
	def print_outputs(self):
		nnum = 0
		for neuron in self.m_layers[-1]:
			nnum += 1
			print "Output for neuron {} is {}".format(nnum, neuron.getValue())
	
	
	def backProp(self):
		done = 0
		if self.using_matrix:
			result = self.backPropagate()
			if(result is 1):
				return 0
			else:
				return 1
		else:
			self.backPropagate()
			
	
	def finished_training(self, errList):
		#given list of errors, loop over and check that all are below the threshold
		cont = 1
		for error in errList:
			if (error > self.thresh):
				return True
		return False
			
	
	def bp_matrix(self):
		pass
		#get error/gradient
		row, col, layer, x = 0, 0, 0, 0
		error, out_list, miss = [], [], []
		
		# loop over neurons in last layer and get values, or use the output matrix.
		for neuron in self.m_layers[layer]:
			out_list.append(neuron.getValue())
			if(self.testing):
				print("out_list = {}, Neuron.getValue returns {}".format(out_list, neuron.getValue()))
			miss.append(self.target_values[x] - neuron.getValue())
			if(self.testing):
				print("Miss = {}".format(miss))
		for val in miss:
			if(self.testing):
				print("val is {}".format(val))
			error.append((sig(self.matrix_outputs[x], deriv=True))*miss[x])
			if(self.testing):
				print("Calculated Error of {}".format(error))
			x += 1
			assert(x <= len(self.matrix_outputs))
		x = 0
		temp_matrix = []
		new_weights = []
		w_layer = -1 #weight layer
		this_layern = -2 #last hidden layer
		
		#set the matrix size.
 		for neuron in self.m_layers[this_layern]: #start at the last hidden layer
			if(neuron is None or abs(this_layern) >= len(self.m_layers)):
				break
			#we loop over all the weights for each neuron, which equals number of nodes in this layer
			nextLayerNodes = len(self.m_layers[this_layern + 1]) #output layer nodes on first pass
			thisLayerNodes = len(self.m_layers[this_layern])
			temp_matrix = zero_matrix(nextLayerNodes, thisLayerNodes)
			
			i, j = 0, 0
			
			#Loop over number of connections (len(last hidden layer))
			for weight in self.matrix_weights[w_layer]: # Start at last matrix and go back.
				if(abs(w_layer) >= self.num_weight_layers):
					break
				#print("weight in matrix_weights[w_layer] = {}".format(weight))
				newWeight = weight + self.eta*error[x]*neuron.getValue()
				self.matrix_weights[w_layer] += self.eta*error[x]*neuron.getValue()
				#temp_matrix[i][j] = newWeight
				
				#print("Temp_matrix = {}".format(temp_matrix))
				x += 1
				w_layer -= 1 # Go back one matrix
				i += 1
				
			this_layern -= 1 # Go back one layer
			
		#set the new weights.
		self.matrix_weights = temp_matrix
		print("Matrix_weights = {}".format(self.matrix_weights))
		
	
	def backPropagate(self):
		"""
		Returns 0 on early exit, else returns 1
		"""
				
		done = 0
		# Get the output errors.
		outErr = []
		outErr = self.get_output_errors()
		if(self.testing):
			print("Output Errors: {}".format(outErr))
		
		i = 0
		#set the Errors in output Neurons.
		#keep track of total error
		total_err = 0.
		
		for n in self.m_layers[-1]:
			n.set_grad(outErr[i])
			total_err += outErr[i]
			i += 1
		
		if abs(total_err) <= self.thresh:
			return 0
			
		print "total output Error = {}".format(total_err)
		i = 0	
		#check that we can get the gradient from each neuron.
		#for n in self.m_layers[-1]:
			#print("Gradient in node is = {}".format(n.get_grad()))
		
		#Now get the hidden layer errors.
		layer_number = -1
		hidden_layers = self.max_layer - 2
		hiddenErr = [[] for i in xrange(hidden_layers)]
		wMatrix = self.matrix_weights
		
			
		for i in range(hidden_layers, 0, -1):
			#hiddenErr[i-1] = np.array(self.get_hidden_errors(wMatrix[i], i))
			hiddenErr[i-1] = np.array(self.get_hidden_errors(wMatrix[i], i))
			
		if(self.testing):
			print("Hidden Errors: {}".format(hiddenErr))
			
					
		#check if the hidden errors are below a certain amount.
		#TODO
		ind = 0
			
		#update weights
		assert(len(hiddenErr) == self.max_layer - 2)
			
			
		for layr in range(0, self.num_weight_layers):
			#output layer
			if(layr == self.num_weight_layers):
				wvals = []
				outputs = np.array([n.getValue() for n in self.m_layers[layr]])
				outputs = np.reshape(outputs, (len(outputs), 1) )
				Oerr = np.reshape(outErr, (1, (len(outErr))))
				if(self.testing):
					print "outputs (used for changing weights) = {}".format(outputs)
				wvals = np.dot(outputs, Oerr) * self.eta
				self.matrix_weights[layr] += wvals.T					
			else:
				wvals = []
				outputs = np.array([n.getValue() for n in self.m_layers[layr]])
				outputs = np.reshape(outputs, (len(outputs), 1))
				err = np.array([n.get_grad() for n in self.m_layers[layr+1]])
				err = np.reshape(err, (1, len(err)))
				if(self.testing):
					print "outputs (used for changing weights) = {}".format(outputs)
				wvals = np.dot(outputs, err) * self.eta
				#wvals = lin_sum(outputs, hiddenErr[layr], True)
				if(self.testing):
					print "wvals for layer {} = {}".format(layr, wvals)
				self.matrix_weights[layr] += wvals.T
						
		if(self.testing):
			print "matrix weights updated:\n" + "*"*25+"\n{}".format(self.matrix_weights)
			
		return 1	
			
	
	def get_output_errors(self):
		errs = []
		total_err = 0
		i = 0
		for n in self.m_layers[-1]:
			val = n.getValue()
			errs.append(sig(val, True) * (self.target_values[i] - val))
			n.set_grad(errs[i])
			total_err += errs[i]
			if(self.testing):
				print("Output error in Neuron {} = {}".format(i, errs[i]))
			i += 1
		if(total_err < self.thresh):
			self.trained = True
		return errs
		
		
	def get_hidden_errors(self, weights, layer):
		#Accept one matrix, and the errors. Get outputs from the next layer.
		wMatrix = weights
		hErr = []
		self.setLayers(layer)
		assert(self.nextLayer is not None)
			
		if(self.testing):
			print("wMatrix = {}".format(wMatrix))
		
		vals, values = [], []
		vals = [n.get_grad() for n in self.nextLayer]
		values = np.array(vals)
			
		if(self.testing):
			print "values is : {}".format(values)
		values = np.reshape(values, (len(values), 1))	
		grad = np.dot(values.T, wMatrix)
			
		if(self.testing):
			print("grad successfully calculated and is {}".format(grad))
						
		
		i = 0
		hErr = np.zeros((1, len(self.thisLayer)))
		
		if(self.testing):
			print "hErr instantiated as: {}".format(hErr)
		
		#using grad, mult each by output of each neuron.
		for neuron in self.thisLayer:
			if(self.testing):
				print "Calculating error using values\nneuron Val = {}\ngrad[i] = {}".format(neuron.getValue(), grad.item(i))
			hErr[0][i] = (sig(neuron.getValue(), deriv=True) * grad.item(0, i))
			if(self.testing):
				print "hErr[i] = {}".format(hErr.item(0, i))
			neuron.set_grad(hErr.item(0, i))
			i += 1
		
		#assert(len(hErr) == len(self.thisLayer))
		if(self.testing):
			print "hErr = {}".format(hErr)	
		
		return hErr
			
	
	def set_weights(self, w):
		self.matrix_weights = w
	
	
	def store_weights(self):
		np.save(path + self.weight_file_name, self.matrix_weights)
		
		
	def load_weights(self):
		#with open(self.weight_file_name, 'rb') as wf:
		#	self.matrix_weights = np.loadtxt(wf)
			
		#print "Loaded weights from file successfully..."
		
		self.matrix_weights = np.load(path + self.weight_file_name + '.npy')
		
			
	def train(self, times):
		count = 0
		for x in range(times):
			self.feedForward()
			c = self.backPropagate()
			count += 1
			if(c == 0):
				break
		self.store_weights()
		return count

			
	def train_test(self, times, bp_type=1):
		count = 0
		
		while(count < times):
			#self.ffMatrix()
			self.feedForward()
			if(bp_type == 1):
				self.backPropagate()
			elif(bp_type == 2):
				self.bp_matrix()
			count += 1
		return count

	
	def get_weights(self):
		return self.matrix_weights
		
		
