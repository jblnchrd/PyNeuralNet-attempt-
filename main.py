"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
12 December 2016
Author: Jacob T. Blanchard
"""

import numpy as np
import random as rand
import time
from math import *

"""
Credit to Dave Miller for some ideas about initializing the network using topology. He has a great video designing a simple
feedforward neural net in c++ at https://www.youtube.com/watch?v=KkwX7FkLfug

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
	"""
	The weights are always in the next layer from first hidden node to output layer.
	
	Calculations involving the dot product or multiple weights and values
	can be carried out at the node itself.
	"""
	def __init__(self, layerNumber, _inputs, val=None, firstLayer=False):
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
		#initialize weights and attach connections to each neuron for each output.
		#weights are in the neuron in the NEXT LAYER. i.e. From first hidden layer to last layer.
		
		if firstLayer is False:
			self.layer_num = layerNumber
			for inp in xrange(self.inputs): #inputs to this neuron.
				r = 2*rand.random() - 1
				self.weights.append(r)
				print("weight %f attached to Neuron" % r)
				
		if val is not None:
			self.value = val
			print("value of %f attached to Neuron" % val)
	
	
	def sumDeriv(self, layer):
		sum = 0
		for neuron in layer:
			for w in neuron.getWeights():
				sum += w*neuron.get_grad()
		return sum
		
	
	def sumWeightVals(self, layer):
		pass
		"""
		#layer is a layer of neuron objects
		s, i = 0, 0
		for weight in self.weights:
			s += weight*(val from incoming neuron)
			#s+=weight*
			"""
		
		
	def setWeight(self, val, index):
		self.weights[index] = val
	
	
	def set_grad(self, val):
		self.gradient = float(val)
	
	
	def get_grad(self):
		return self.gradient	
		
	
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
	""" Neural Network Implementation
	
		Note: The weights can be created and accessed in various ways. A connection
		object can store the weights, they can be stored in matrices in the usual way,
		or we can directly access weights initiated and stored in a neuron object.
		The weight stored in a neuron exists in the first hidden layer to the last layer (output).
		So, when calculating with these weights, they must be used to multiply values that
		are INCOMING to the neuron. Thus, the weights for connections from some layer
		A to a layer B are stored in the neurons in the B layer. Calculate accordingly,
		looping over each layer and using the next layer to access the weights.
	
	"""
	def __init__(self, topology, inputs, matrix=False, rate=0.5, test=False):
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
		self.layerNumber = 0
		self.error_vectors = []
		#initialize weights in a matrix.
		# we'll have len(topology) - 1 matrices in this list.
		self.matrix_weights = [[] for x in xrange(self.max_layer - 1)] 
		self.matrix_hiddens = []
		self.eta = rate
		self.vector_grad = [[] for x in xrange(self.max_layer)]
		self.val_vector = [[] for x in xrange(len(topology))] # len(topology) value vectors in one matrix.
		self.testing = test
			
		#initialize the list of lists with the number of sub-lists == length of topology
		self.m_layers = [[] for i in xrange(len(topology))]
		self.val_vector[0] = np.array(inputs).T
		
		#attach input Neurons...
		for x in range(self.num_inputs):
			self.m_layers[0].append(Neuron(layerNumber=0, _inputs=1, val=inputs[x], firstLayer=True))
			print("attached input Neuron in first layer")
			
		#attach hidden Neurons using topology....Create a new connection
		for layer in range(1, len(topology) - 1):
			for n in range(topology[layer]):
				#need one new connection per node.			
				if layer + 1 < len(topology): #not last hidden layer
					self.m_layers[layer].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False))
					self.layerNumber = layer
					print(("attached hidden Neuron in layer number %d") % (layer + 1))
				else: #last hidden layer
					self.m_layers[-2].append(Neuron(layerNumber=layer, _inputs=topology[layer-1], firstLayer=False))
					print(("attached hidden Neuron in layer number %d") % (layer + 1))
					self.layerNumber = layer
		
		#attach output Neurons...
		for x in range(self.num_outputs):
			self.m_layers[-1].append(Neuron(layerNumber=self.max_layer, _inputs=topology[-2], firstLayer=False))
			self.layerNumber = len(topology)
			print("attached output Neuron in last layer")
			
		#Set up weights if using matrices		
		if(matrix):
			#Store inputs in a numpy vector
			self.val_vector[0] = self.matrix_inputs
			
			# setting up the matrix holding all weights, with x being the (layer - 1)st connection.
			for x in range(1, len(topology)):
				self.matrix_weights[x-1] = 2 * np.random.random((topology[x], topology[x-1])) - 1
				
			if(self.testing):
				print("-"*20 + "MATRIX WEIGHTS" + "-"*20)
				print(self.matrix_weights)
			
			#create input matrix
			self.matrix_inputs = np.array(self.input_list).T
			
			if(self.testing):
				print("-"*20 + "INPUT MATRIX" + "-"*20)
				print(self.matrix_inputs)		
						
		
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
					#self.connections[this_layer - 1][i].setWeight(w[i]) # For use in backPropagation
				valp = neuron.processFunction(sum, function="sigmoid", deriv=False)
				neuron.setValue(valp)
				sum = 0
				valp = 0
				w = []
				values = []
		
		if(self.testing):
			self.print_outputs()	
	
	
	def feedForward_Matrix(self):
		#set up the layers
		self.nextLayer = self.m_layers[1]
		self.prevLayer = self.m_layers[0]
		ind, layer, this_layer, this_node = 1, 0, 0, 0
		
		for x in xrange(0, self.num_weight_layers ):
			
			inp = np.array(self.matrix_inputs)
			weight_matrix = np.array(self.matrix_weights[x])
			next_matrix = sig(np.dot(weight_matrix, inp))
			self.setMatrixInputs(next_matrix)
			self.val_vector[ind] = next_matrix
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
			
				
	def print_outputs(self):
		for neuron in self.m_layers[-1]:
			out = neuron.getValue()
			if(self.testing):
				print("Output for neuron is {}".format(out))
	
	
	def backProp(self):
		if self.using_matrix:
			self.backPropagate()
		else:
			self.backPropagate()
			
	
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
			print("Temp_matrix initialized to {}".format(temp_matrix))
			i, j = 0, 0
			
			#Loop over number of connections (len(last hidden layer))
			for weight in self.matrix_weights[w_layer]: # Start at last matrix and go back.
				if(abs(w_layer) >= self.num_weight_layers):
					break
				print("weight in matrix_weights[w_layer] = {}".format(weight))
				newWeight = weight + self.eta*error[x]*neuron.getValue()
				self.matrix_weights[w_layer] += self.eta*error[x]*neuron.getValue()
				#temp_matrix[i][j] = newWeight
				
				print("Temp_matrix = {}".format(temp_matrix))
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
	
	
	def backPropagate(self):
			# Print Print Print....
			#Set up the error matrix.
			# For each node, create a matrix entry.
			errMatrix = []
			for layer in self.m_layers:
				errMatrix.append([])
			"""
			In detail.
			1. Get the miss, that is, target - output for each output neuron
			2. Use this, along with the sigmoid(output) to get the gradient or error miss*sig(output)
			3. Attach these values to the appropriate neurons
			4. Get the hidden errors, moving backward through the net.
			5. Use the error in the output neurons to calculate the error/gradient in the hidden layers
			6. Use the errors calculated to update the weights.
			7. Repeat
			"""
			# Get the output errors.
			outErr = []
			outErr = self.get_output_errors()
			print("Output Errors: {}".format(outErr))
			
			i = 0
			#set the Errors in output Neurons.
			for n in self.m_layers[-1]:
				n.set_grad(outErr[i])
				i += 1
				
			#check that we can get the gradient from each neuron.
			for n in self.m_layers[-1]:
				print("Gradient in node is = {}".format(n.get_grad()))
			
			#Now get the hidden layer errors.
			layer_number = -1
			hidden_layers = self.max_layer - 1
			hiddenErr = [[] for i in xrange(hidden_layers)]

			i = -1
			wMatrix = self.matrix_weights
			while(abs(i) <= len(self.matrix_weights)):
				#wMatrix = self.matrix_weights[i]	
				#print("wMatrix = {}".format(wMatrix))
				hiddenErr[i] = np.array(self.get_hidden_errors(wMatrix[i], i))
				i -= 1
			print("Hidden Errors: {}".format(hiddenErr))
			
			#Errors are set in the nodes, so we can put them in a matrix and move forward.
			error = [[] for i in xrange(self.max_layer)]
			i = self.max_layer
			j = 0
			
			while(i > 0 and i < self.max_layer):
				error[j] += [self.m_layers[i].get_grad()]
				print("Errors:{}".format(self.m_layers[i].get_grad()))
				i -= 1	
				j += 1
				print("Error[j] = ".format(error[j-1]))
			#print("Errors in order : {}".format(error))
							
	
	
	def get_output_errors(self):
		errs = []
		i = 0
		for n in self.m_layers[-1]:
			errs.append(sig(n.getValue(), True) * (self.target_values[i] - n.getValue()))
			i += 1
		return errs
		
		
	def get_hidden_errors(self, weights, layer):
		#Accept one matrix, and the errors. Get outputs from the prevLayer.
		thisLayer = self.m_layers[layer]
		prevLayer = self.m_layers[layer-1]
		this_size = len(thisLayer)
		prev_size = len(prevLayer)
		wMatrix = weights
		print("wMatrix = {}".format(wMatrix))
		vals, values = [], []
			
		#put all the errors from this layer in a list.
		i = 0
		for neuron in thisLayer:
			if(neuron.get_grad() is not None):	
				vals.append(neuron.get_grad())
		
		if(vals is not None):
			values = np.array(vals)
			print("vals is : {}".format(values))
			grad = np.dot(values, wMatrix)
			print("grad successfully calculated and is {}".format(grad))
			#return grad
			#check if the previous layer is null
		
		if(prevLayer is not None):
			i = 0
			hErr = []
			#using grad, mult each by output of each neuron.
			for neuron in prevLayer:
				hErr.append(sig(neuron.getValue(), deriv=True)*grad[i])
				neuron.set_grad(hErr[i])
				i += 1
			print("Gradients are {}".format(hErr))
			return hErr
			
			
	def train(self, times):
		for x in range(times):
			self.feedForward()
			self.backProp()


def lin_sum(listA, listB):
	z = 0
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
		
#main
if __name__ == "__main__":
	rand.seed(1)
	print("Welcome to my custom neural network!")
	print("-"*25 + "WARNING" + "-"*25)
	print("User retard level must be below 10 to use this program.\n\n")
	net_top = np.array([2, 3, 2])
	top = [2, 3, 2] # topology specifies the number of layers and nodes in each layer.
	#specify inputs and outputs. Then the topology can be used
	# For example, [2, 1, 2] means 2 inputs, 1 hidden node, and 2 outputs.
	inps = [0.332, -0.126]
	targets = [1, 0]
	myNet = Net(top, inps, matrix=True, test=True)
	print("\n")
	myNet.feedForward()
	myNet.setTargetValues(targets)
	myNet.print_outputs()
	myNet.backProp()
