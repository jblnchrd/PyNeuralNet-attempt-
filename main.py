import numpy as np
import random as rand
import time
from math import *
from tempfile import TemporaryFile
import os

"""
Update: Need a way to keep track of which layer the neurons are in if the order of their 
instantiation isn't kept. It may not matter, however, and this needs to be determined.
Update: The order is being kept by using a list of lists. The topology encodes the entire
network, and the index of each sublist represents the layer for the neurons.

Update December 6 2016: We can use numpy matrices for the calculations which would probably 
be easier conceptually and programatically to implement. 
There would be a weight matrix for each connection between layers, with the number of rows being equal 
to the number of neurons in the layer holding these weights. The columns equal the number of 
neurons in the previous layer.
To feedForward the network, we simply multiply the input matrix (column vector) by the weight matrix.
We get the values for each neuron based on the output vector. These are attached to the hidden layer
neurons and used for the inputs for the next calculation.

X = inputs.
W_1 = weights from inputs to first hidden layer. Weights[next_layer]

Update May 16, 2017: The matrix form of the feedforward member function of the Net class needs to store the values
of the first calculation (first weight matrix times input matrix) in the hidden layer, but first we can
simply bypass storing them in the neuron objects while we get the calculation right. These values must
be set so that we can use these values to back propagate the network.

Update May 25, 2017.
Back propagation will be primarily done through objects, that is, node objects with the incoming weights
stored in the neurons themselves. Calculate all the errors first, then use these values to get the
new weights for the network.

"""

	
class Neuron(object):
	"""
	The weights are always in the next layer from first hidden node to output layer.
	
	Calculations involving the dot product or multiple weights and values
	can be carried out at the node itself.
	"""
	def __init__(self, layerNumber, _inputs, val=None, firstLayer=False, testing=False):
		self.inputs = _inputs
		self.weights = []
		self.d_weights = []
		self.value = val
		self.layer_num = layerNumber
		self.is_first = firstLayer
		self.is_last = False
		self.matrixWeights = None
		self.gradient = 0
		self.next_nodes = None
		self.from_nodes = None
		self.testing = testing
		
		#initialize weights and attach connections to each neuron for each output.
		#weights are in the neuron in the NEXT LAYER. i.e. From first hidden layer to last layer.
		
		if firstLayer is False:
			self.layer_num = layerNumber
			for inp in xrange(self.inputs): #inputs to this neuron.
				r = 2*rand.random() - 1
				self.weights.append(r)
				
				
		if val is not None:
			self.value = val
				
	
	def sumDeriv(self, layer):
		sum = 0
		for neuron in layer:
			for w in neuron.getWeights():
				sum += w*neuron.get_grad()
		return sum
		
	
	def sumWeightVals(self, layer):
		pass
		
		
	def setWeight(self, val, index):
		self.weights[index] = val
	
	
	def set_grad(self, val):
		if(type(val) is list or type(val) is np.ndarray):
			self.gradient = val
		else:
			self.gradient = float(val)
	
	
	def get_grad(self):
		return self.gradient	
		
	
	def setMatrixInputs(self, num_inputs, inp_list):
		pass
	
	def setMatrixWeights(self, inputs, outputs):
		self.matrixWeights = 2 * np.random.random((outputs, inputs)) - 1
	

	def getLayerNumber(self):
		return self.layer_num
		
		
	def getInputs(self):
		if self.inputs is not None:
			return self.inputs
		else:
			print("No inputs to this neuron.")
	
	
	def setValue(self, val):
		self.value = val
	
	
	def getValue(self):
		if self.value is not None:
			return self.value
		else:
			print("No value in this Neuron.")
	
	
	def getWeights(self, weightNum = None):
		if(weightNum is None):
			return self.weights
		else:
			return self.weights[weightNum]
	
	
	def sigmoid(self, x, deriv=False):
		if(deriv):
			return x*(1-x)
		return 1/(1 + np.exp(-x))
		
		
	def transferFunction(self, x, deriv=False):
		if(deriv):
			return 1.0 - x**2 
		return tanh(x)
		

	def processFunction(self, x, function="sigmoid", deriv=False):
		if(deriv):
			if(function == "sigmoid"):
				v = self.sigmoid(x, deriv=True)
				return v
			elif(function == "tan"):
				v = self.transferFunction(x, deriv=True)
				return v
		else:
			if(function == "sigmoid"):
				v = self.sigmoid(x, deriv=False)
				return v
			elif(function == "tan"):
				v = self.transferFunction(x, deriv=False)
				return v
		
		return 0 #better not get here


class Layer(object):
	"""
	 Layer holding connection information
	"""
	def __init__(self, nodeList, layernum, inputs, fromSize=0, toSize=0):
		self.prevLayer_size = fromSize
		self.nextLayer_size = toSize
		#Nodes in this layer
		self.nodes = []
		assert(nodeList is not None)
		for node in nodeList:
			#create nodes
			self.nodes.append(Neuron(layernum, inputs))
		
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
	def __init__(self, topology, inputs, matrix=False, rate=0.5, load_=False, weightFile=None, test=False):
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
		self.target_values = []
		self.num_weight_layers = len(topology) - 1
		self.matrix_outputs = []
		self.layerNumber = 0
		self.matrix_weights = [[] for x in xrange(self.max_layer - 1)] 
		self.eta = rate
		self.vector_grad = [[] for x in xrange(self.max_layer)]
		self.testing = test
		self.Error = 0
		self.thresh = 0.0015
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
			np.random.seed(3938)
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
			
			print "Loaded matrix weights:\n{}".format(self.matrix_weights)
			
						
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
	
	
	def ffNormal(self):
		pass
			
	
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
		
	
	def sumLin(self, layerno, errors):
		#layerno is the layer previous to errors.
		suml = np.dot(self.matrix_weights[layerno], errors)
		return suml
	
	
	def run(self):
		ffMatrix()			
	
	
	def backPropagate(self):
			"""
			Returns 1 on early exit, else returns 0
			"""
			# Print Print Print....
			#Set up the error matrix.
			# For each node, create a matrix entry.
			#self.Error /= len(self.m_layers[-1]) - 1
			#self.Error = sqrt(self.Error)
			
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
			
				
			#check that we can get the gradient from each neuron.
			#for n in self.m_layers[-1]:
				#print("Gradient in node is = {}".format(n.get_grad()))
			
			#Now get the hidden layer errors.
			layer_number = - 1
			hidden_layers = self.max_layer - 2
			hiddenErr = [[] for i in xrange(hidden_layers)]
			wMatrix = self.matrix_weights
			
			
			for i in range(hidden_layers, 0, -1):
				#hiddenErr[i-1] = np.array(self.get_hidden_errors(wMatrix[i], i))
				hiddenErr[i-1] = np.array(self.get_hidden_errors(wMatrix[i], i))
			
			if(self.testing):
				print("Hidden Errors: {}".format(hiddenErr))
			
			#check/print gradients in all layers
			#l = 0
			#if(self.testing):
				#for layer in self.m_layers:
					#l += 1
					#ind = 0
					#for neuron in layer:
						#print "Gradients in layer {}, Neuron {} = {}".format(l, ind+1, neuron.get_grad())
						#ind += 1
			
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
			
			return 0	
			
	
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
		np.save('/home/jsurg/Programs/python/' + self.weight_file_name, self.matrix_weights)
		
		
	def load_weights(self):
		#with open(self.weight_file_name, 'rb') as wf:
		#	self.matrix_weights = np.loadtxt(wf)
			
		#print "Loaded weights from file successfully..."
		
		self.matrix_weights = np.load('/home/jsurg/Programs/python/' + self.weight_file_name + '.npy')
		
		print "Matrix weights loaded from file = {}".format(self.matrix_weights)
	
	
	def train(self, times, wFile):
		count = 0
		for x in range(times):
			self.feedForward()
			c = self.backPropagate()
			count += 1
			if(c == 0):
				break
		self.store_weights()
		#record_weights(self.matrix_weights, wFile)
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
		
		
def lin_sum(listA, listB, dotprod=False):
	z = 0
	Z = []
	if(dotprod):
		for x in listA:
			z = 0
			for y in listB:
				z += x*y
				Z.append(z)
		return Z
	
	for x in listA:
		for y in listB:
			z += x*y
			
	return z
		
		
def sig(x, deriv=False):
	if(deriv):
		return x*(1-x)
	return 1/(1 + np.exp(-x))


def rand_matrix(rows, cols):
	return np.random.random(rows, cols) - 1
	
	
def zero_matrix(fromNodes, toNodes):
	return np.zeros(shape(toNodes, fromNodes))


def mult_matrix(A, B):
	if (np.dot(A, B)):
		return np.dot(A, B)
	else:
		print("Wrong dimensions...")
		
		
def record_weights(weights, file_name):
	#convert items in numpy array to regular list.
	data = []
	for weight in weights:
		if weight != "array" and type(weight) is float:
			data.append(weight)
			#print "data is {}".format(weight)
	with open(file_name, 'a+') as f:
		for d in data:
			f.write("{}\n").format(d)


def load_weights(file_name, layers):
	#get the weights from a file.
	weights = [[] for i in range(layers)]
	with open(file_name, 'r') as f:
		lines = f.readlines()
		i, j = 0, 0
		for line in lines:
			for val in line:
				print val
				#weights[i][j] = val
				#j += 1
			#i += 1
	
	return weights
	
	
#main
if __name__ == "__main__":
	rand.seed(1)
	print("Welcome to my custom neural network!")
	print("-"*25 + "WARNING" + "-"*25)
	print("User retard level must be below 10 to use this program.\n\n")
	net_top = np.array([10, 5, 7, 1])
	top = [10, 5, 5, 7, 4] # topology specifies the number of layers and nodes in each layer.
	#specify inputs and outputs. Then the topology can be used
	
	# For example, [2, 1, 2] means 2 inputs, 1 hidden node, and 2 outputs.
	inps = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 1])
	targets = [0., 1., 1., 0.]
	#myNet = Net(top, inps, matrix=True, load_=True, weightFile=fname, test=True)
	myNet = Net(top, inps, matrix=True, load_=False, weightFile="weights", test=True)
	print("\n")
	
	myNet.setTargetValues(targets)
	times = input("Enter number of times to train\n:")
	c = myNet.train(times, "weights.txt")
	print "Ran {} times".format(c)
	
	w = load_weights("weights.txt", len(top))
	myNet_trained = Net(top, inps, matrix=True, load_=True, weightFile="weights", test=True)
	myNet_trained.ffMatrix()
	myNet_trained.print_outputs()
	#record_weights(myNet_trained.get_weights(), "weightstxt.txt")
	print "Neural Net finished successfully..."
