import numpy as np
import random as rand
import time
from math import *

"""
There would be a weight matrix for each connection between layers, with the number of rows being equal 
to the number of neurons in the layer holding these weights. The columns equal the number of 
neurons in the previous layer.
To feedForward the network, we simply multiply the input matrix (column vector) by the weight matrix.
We get the values for each neuron based on the output vector. These are attached to the hidden layer
neurons and used for the inputs for the next calculation.

X = inputs.
W_1 = weights from inputs to first hidden layer. Weights[next_layer]

Right now there are two ways to use the network. Either using matrices without storing any values
inside the Neuron objects (yet), or using the neurons without matrices, which seems a bit too complicated than it needs to be.
Just use the matrices, it's way easier.
"""

	
class Neuron(object):
	
	def __init__(self, layerNumber, _inputs, val=None, firstLayer=False):
		self.inputs = _inputs
		self.weights = []
		self.d_weights = []
		self.value = val
		self.layer_num = 0
		self.is_first = firstLayer
		self.is_last = False
		self.matrixWeights = None
		
		#initialize weights and attach connections to each neuron for each output.
		if firstLayer is False:
			self.layer_num = layerNumber
			for input in xrange(self.inputs):
				r = 2*rand.random() - 1
				self.weights.append(r)
				print("weight %f attached to Neuron" % r)
	
		if val is not None:
			self.value = val
			print("value of %f attached to Neuron" % val)
	
	
	def setMatrixInputs(self, num_inputs, inp_list):
		for inp in inp_list:
			pass

	def setMatrixWeights(self, inputs, outputs):
		self.matrixWeights = 2*np.random.random((outputs, inputs)) - 1
	

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
	
	
	def getWeights(self):
		return self.weights
		
		
	def feedForward_N(self, prevLayer):
		pass
		
	
	def feedForward_Matrix(self):
		pass


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


		
class Net(object):
	""" Neural Network Implementation"""
	def __init__(self, topology, inputs, matrix=False):
		self.num_inputs = topology[0]
		self.num_outputs = topology[-1]
		self.using_matrix = matrix
		assert(len(inputs) == self.num_inputs)
		self.nextLayer = None
		self.prevLayer = None
		self.max_layer = len(topology)
		self.input_list = [inputs]
		self.matrix_inputs = np.array([self.input_list])
		self.target_values = []
		self.num_weight_layers = len(topology) - 1
		self.matrix_outputs = []

		#initialize weights in a matrix.
		# we'll have len(topology) - 1 matrices in this list.
		self.matrix_weights = [[] for x in xrange(self.max_layer - 1)] 

		#initialize the list of lists with the number of sub-lists == length of topology
		self.m_layers = [[] for i in xrange(len(topology))]
		
		#attach input Neurons...
		for x in range(self.num_inputs):
			self.m_layers[0].append(Neuron(layerNumber=0, _inputs=1, val=inputs[x], firstLayer=True))
			print("attached input Neuron in first layer")
			
		#attach hidden Neurons using topology....
		for layer in range(1, len(topology) - 1):
			for n in range(topology[layer]):
							
				if layer + 1 < len(topology): #not last hidden layer
					self.m_layers[layer].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False))
					print(("attached hidden Neuron in layer number %d") % (layer + 1))
				else: #last hidden layer
					self.m_layers[-2].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False))
					print(("attached hidden Neuron in layer number %d") % (layer + 1))
		
		#attach output Neurons...
		for x in range(self.num_outputs):
			self.m_layers[-1].append(Neuron(layerNumber=self.max_layer, _inputs=topology[-2], firstLayer=False))
			print("attached output Neuron in last layer")
				
		if(matrix):
			for x in range(1, len(topology)):
				self.matrix_weights[x-1] = np.random.random((topology[x], topology[x-1])) - 1
			print("-"*20 + "MATRIX WEIGHTS" + "-"*20)
			print(self.matrix_weights)
			#create input matrix
			self.matrix_inputs = np.array(self.input_list).T
			print("-"*20 + "INPUT MATRIX" + "-"*20)
			print(self.matrix_inputs)		


	def setTargetValues(self, targs):
		assert(len(targs) == self.num_outputs)
		self.target_values = targs


	def printNodes(self):
		for layer in self.m_layers:
			for neuron in layer:
				print neuron
				
	def feedForward(self):
		if(self.using_matrix):
			self.feedForward_Matrix()
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
		self.print_outputs()	
	
	
	def feedForward_Matrix(self):
		#inputs is a vector.
		#multiply the input and weight matrices.
		ind = 0
		for x in xrange(0, self.num_weight_layers):
			l0 = np.array(self.matrix_inputs)
			l1 = np.array(self.matrix_weights[x])
			next_matrix = sig(np.dot(l1, l0))
			for layer in self.m_layers:
				ind = 0
				for neuron in layer:
					if neuron.layerNumber == 0:
						continue
					elif neuron.layerNumber > 0:
						#set the values.
						assert(len(next_matrix) == len(layer))
						neuron.setValue(next_matrix[ind])
						ind += 1
			#set values for next matrix (inputs).
			self.matrix_outputs = next_matrix
			print("Output Matrix = {}".format(next_matrix))

	def getError(self):
		pass


	def print_outputs(self):
		for neuron in self.m_layers[-1]:
			out = neuron.getValue()
			#print("Output for neuron is %f" % out)
			print("Output for neuron is {}".format(out))
	
	
	def backProp(self):
		#get the error. δ=(t-o)(1-o)o
		pass
	
	def sumlin(self, vals, weights): #vals: list of values. weights: list of weights.
		for comp in xrange(self.num_outputs):
			pass
	
	def train(self, times):
		pass

		
def sig(x, deriv=False):
	if(deriv):
		return x*(1-x)
	return 1/(1 + np.exp(-x))


#main
if __name__ == "__main__":
	rand.seed(time.time())
	print("Welcome to my custom neural network!")
	print("-"*25 + "WARNING" + "-"*25)
	print("User retard level must be below 10 to use this program.")
	net_top = np.array([2, 3, 2])
	top = [2, 3, 2] # topology specifies the number of layers and nodes in each layer.
	#specify inputs and outputs. Then the topology can be used
	# For example, [2, 1, 2] means 2 inputs, 1 hidden node, and 2 outputs.
	inps = [1.232, -0.126]
	targets = [1, 0]
	myNet = Net(top, inps, matrix=True)
	print("\n\n")
	#myNet.feedForward()
	myNet.feedForward()
	myNet.setTargetValues(targets)

	#myNet.printNodes()
	
	#Net.train(100)
	#outputs = Net.getOutputs()
	#print(outputs) 1

	#[5,14,22,14,7]