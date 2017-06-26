import numpy as np
import random as rand
import time
from Neuron import *
import math
import os
import copy 


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
	
	def __init__(self, topology, inputs, targs, matrix=True, rate=0.35, threshold=0.001, load_=False, weightFile=None, test=False, func="sig"):
		self.function = func
		self.num_inputs = topology[0]
		self.num_outputs = topology[-1]
		self.using_matrix = matrix
		assert(len(inputs[0]) == self.num_inputs)
		self.nextLayer = None
		self.prevLayer = None
		self.thisLayer = None
		self.max_layer = len(topology)
		self.input_list = inputs
		self.matrix_inputs = np.array([self.input_list]).T
		self.target_values = targs[0]
		self.target_list = targs
		self.num_weight_layers = len(topology) - 1
		self.layerNumber = 0
		self.matrix_weights = [[] for x in xrange(self.max_layer - 1)] 
		self.eta = rate
		self.testing = test
		self.thresh = threshold
		self.trained = False
		self.vectors = [[] for x in range(len(topology))] # one list of values for each layer
		self.load = load_
		self.weight_file_name = weightFile
		self.input_length = len(inputs)
		self.average_err = [1. for i in range(self.input_length)]
		self.total_error = 0.0
		self.input_number = 0
		self.recent_avg_smoothing_factor = 100.0
		self.recent_avg_error = 0.0
		self.total_errors = [1 for i in range(self.input_length)]
		self.hidden_errors = [[] for i in range(self.max_layer - 2)]
		self.output_errors = []
		self.all_errors = []
		#initialize the list of lists with the number of sub-lists == length of topology
		self.m_layers = [[] for i in xrange(len(topology))]
		
		#attach input Neurons...
		for x in range(self.num_inputs):
			self.m_layers[0].append(Neuron(layerNumber=0, _inputs=1, val=self.input_list[0][x], firstLayer=True, testing=test))
			self.vectors[0].append(inputs[0][x])
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
		assert(len(targs[0]) == self.num_outputs)
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
		self.setLayers(this_layer)
			
		for layer in self.m_layers:
			if this_layer == self.max_layer:
				break
			this_layer += 1
			self.setLayers(this_layer)
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
				valp = neuron.processFunction(x=sum, function="sigmoid", deriv=False)
				neuron.setValue(valp)
				sum = 0
				valp = 0
				w = []
				
		
		if(self.testing):
			self.print_outputs()	
	
	
	def backProp_Normal(self):
		pass
		# get output errors
		#get hidden layer errors
		#update weights
		
	
	def ffMatrix(self):
		index, wLayer = 1, 0,
		layer = 0
		self.setLayers(layer)
		num_Hlayers = self.max_layer - 1
		vals = []
		#calculate all hidden layer values (and outputs)
		for layer in range(0, self.max_layer - 1):
			index = 0
			self.setLayers(layer) #layer started at 0
			weights = np.array(self.matrix_weights[layer])
			
			if(self.testing):
				print "weights = {}".format(weights)
				print "vectors[{}] = {}".format(layer, self.vectors[layer])
			if(self.function == "sig"):
				next_matrix = sig(weights.dot(self.vectors[layer]))
			elif(self.function == "tan"):
				next_matrix = math.tan(weights.dot(self.vectors[layer]))
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
		
		
	def print_outputs(self):
		nnum = 0
		for neuron in self.m_layers[-1]:
			nnum += 1
			print "Value {}: {}".format(nnum, neuron.getValue())
	
	
	def backProp(self):
		if self.using_matrix:
			self.backPropagate()
		else:
			self.backPropagate()
			
	
	def finished_training(self):
		#given list of errors, loop over and check that average are below the threshold
		for e in range(self.input_length):
			if(self.average_err[e] > self.thresh):
				self.input_number = e #Return to previous input set to train
				return False # Return False on first occurence
			elif(self.average_err[e] <= self.thresh):
				if(e == self.input_length):
					return True
				else:
					continue
	
			
	def get_avg_err(self):
		return self.average_err
		
	
	def get_avg_err_input(self, inp):
		return self.average_err[inp]
		
	def mutate(self):
		#mutate (change) a random weight and check if errors are better
		pass
	
	
	def kill_node(self, layer, node):
		# must set a number 
		pass	
	
	
	def backPropagate(self):
		"""
		Returns 0 on early exit, else returns 1
		"""	
		# Get the output errors.
		outErr = []
		i, count, layer_number, hidden_layers = 0, 1, -1, self.max_layer - 2
		outErr = self.get_output_errors()
				
		if(self.testing):
			print("Output Errors: {}".format(outErr))
	
		#calculate and store the average error for this input
		assert(len(outErr) == len(self.m_layers[-1]))
				
		#Now get the hidden layer errors.
		hiddenErr = [[] for i in xrange(hidden_layers)]
		wMatrix = self.matrix_weights
					
		for i in range(hidden_layers, 0, -1):
			hiddenErr[i-1] = np.array(self.get_hidden_errors(wMatrix[i], i))
			
		if(self.testing):
			print("Hidden Errors: {}".format(hiddenErr))
			
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
				if(self.testing):
					print "wvals for layer {} = {}".format(layr, wvals)
				self.matrix_weights[layr] += wvals.T
						
		if(self.testing):
			print "matrix weights updated:\n" + "*"*25+"\n{}".format(self.matrix_weights)
			
		return 1	
		
	def get_deltas(self):
		delta = []
		i = 0
		for n in self.m_layers[-1]:
			val = n.getValue()
			delta[i] = abs(self.target_values[i] - val)
			i += 1
		
		return delta
	
	
	def get_output_errors(self):
		errs = []
		delta, total_err, i, error = 0., 0., 0, 0.
		miss = 0.
		e = 0.
		for n in self.m_layers[-1]:
			val = n.getValue()
			delta = (self.target_values[i] - val) ** 2
			miss = self.target_values[i] - val
			error += delta ** 2
			if self.function == "sig":
				e = n.processFunction(deriv=True)
				#errs.append(sig(val, True) * (self.target_values[i] - val))
				errs.append(e * miss)
			elif self.function == "tan":
				e = n.processFunction(function="tan", deriv=True) * miss
				errs.append(e)
			n.set_grad(errs[i])
			if(self.testing):
				print("Output error in Neuron {} = {}".format(i, errs[i]))
			i += 1
		if error <= self.thresh:
			self.trained = True
			
		#Error handling
		error /= len(self.m_layers[-1])
		error = math.sqrt(error)
		self.total_error = error
		self.total_errors[self.input_number] = abs(error)
		
		# Recent average measurement
		recent_avg_err = 0.0
		recent_avg_err = (recent_avg_err * self.recent_avg_smoothing_factor + error) / (self.recent_avg_smoothing_factor + 1.0)
		self.recent_avg_error = recent_avg_err
		self.output_errors = errs
		return errs
		
	
	def set_total_errors(self):
		total_error
		errors = self.get_output_errors()
		for err in errors:
			total_error += abs(err)
		return total_error 
			
	def get_total_errors(self):
		return self.total_errors
		
	def get_recent_average_error(self):
		return self.recent_avg_error
		
		
	def get_sum_errors(self):
		s = np.sum(self.average_err)
		return s
	
	
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
			if(self.function == "sig"):
				hErr[0][i] = (sig(neuron.getValue(), deriv=True) * grad.item(0, i))
			elif(self.function == "tan"):
				#hErr[0][i] = (sig(neuron.getValue(), deriv=True) * grad.item(0, i))
				hErr[0][i] = (neuron.processFunction(function="tan", deriv=True) * grad.item(0, i))
			if(self.testing):
				print "hErr[i] = {}".format(hErr.item(0, i))
			neuron.set_grad(hErr.item(0, i))
			i += 1
		
		#assert(len(hErr) == len(self.thisLayer))
		if(self.testing):
			print "hErr = {}".format(hErr)	
		self.hidden_errors = hErr
		return hErr
			
	
	def set_weights(self, w):
		self.matrix_weights = w
	
	
	def store_weights(self):
		np.save(path + self.weight_file_name, self.matrix_weights)
		
		
	def load_weights(self):	
		self.matrix_weights = np.load(path + self.weight_file_name + '.npy')
		
	
	def getOutputs(self):
		return [n.getValue() for n in self.m_layers[-1]]
		
		
	def batch_train(self, times, tolerance):
		num_inputs = len(self.input_list)
		count = 0
		for run in range(times):
			for x in range(num_inputs):
				self.set_inputs(x)
				self.feedForward()
				self.backPropagate()
				count += 1
			os.system('clear')
			print "Run: {}".format(count)
			print "Total Error:\t\t{}".format(self.total_error)
			print "Average Errors:\t\t{}".format(self.average_err)
			print "Recent Avg Err:\t\t{}".format(self.recent_avg_error)
		
		self.store_weights()
	
	
	def set_inputs(self, num):
		self.input_number = num
		self.vectors[0] = np.array(self.input_list[num])
		self.target_values = np.array(self.target_list[num])
		
	# 1 in net is trained, 0 if switching 
	def check_errors(self):
		if self.total_errors[self.input_number] is None or self.total_errors[self.input_number] == 0:
			return 0
		if(self.net_trained()):
			return 1
		for e in range(self.input_length):
			if self.total_errors[e] > self.thresh:
				self.input_number = e
				self.set_inputs(self.input_number)
				return 0
			elif e <= self.input_length:
				if self.net_trained():
					return 1
				else:
					self.input_number += 1
					self.set_inputs(e)
					return 0
		return -1
		
	
	def net_trained(self):
		#loop over all input sets and check that all are trained.
		for i in range(self.input_length):
			if(self.total_errors[i] > self.thresh):
				return False
		return True
	
	def composite_train(self):
		pass
		#create a new network with the same parameters, but slightly different weights and nodes.
		#Compare this network's errors with the original, using whichever weights are better.
		
	def get_weights(self):
		return self.matrix_weights	
	
	def train(self, times=10000):
		self.set_inputs(0)
		self.feedForward()
		self.backProp()
		
		for setno in range(self.input_length):
			self.total_error = 0
			self.set_inputs(setno)
			for epoch in range(times):
				self.feedForward()
				self.backPropagate()
				if self.total_errors[self.input_number] <= self.thresh:
					break
		
		self.store_weights()
	
	
	def print_stats(self):
		print("Training Input Set Number: {}".format(self.input_number))
		print("Total Error: {}".format(self.total_error))
		print("Error for this set: {}".format(self.total_errors[self.input_number]))
		print "Recent average error: {}".format(self.recent_avg_error)
		
		
	def simple_train(self, times):
		count = 0
		print "Training. This may take a while"
		for y in range(times):
			for x in range(len(self.input_list)):
				self.input_number = x
				self.vectors[0] = np.array(self.input_list[x])
				self.target_values = np.array(self.target_list[x])
				self.feedForward()
				self.backPropagate()
				count += 1
		self.store_weights()
		return count
	
	def print_inputs_f(self):
		inputs = self.input_length
		for i in self.input_list:
			print i
	
	def predict(self, input_vals, targets):	
		print "Make sure your targets are correct!\n"
		print "Predicting ..."
		self.load_weights()
		vals = []
		int_vals = []
		i = 0
		for x in range(len(input_vals)):
			self.vectors[0] = np.array(input_vals[x])
			self.target_values = np.array(targets[x])
			self.feedForward()
			int_vals += [int(round(n.getValue())) for n in self.m_layers[-1] ]
			vals += [n.getValue() for n in self.m_layers[-1] ]
		print "Given inputs "
		self.input_list = input_vals
		self.print_inputs_f()
		print "Net predicts: {}.\nTargets: {}".format(int_vals, targets)
		print "Net predicts: {}.".format(vals)
	
		
	# WORKING: Error terminated only. Trains all inputs simultaneously.	
	def Train(self):
		self.input_number = 0
		self.set_inputs(0)
		netTrained = False
		count = 0
		print "Training. This may take a while"
		self.set_inputs(0)
		while netTrained == False:
			for x in range(250):
				self.set_inputs(x % self.input_length)
				self.feedForward()
				self.backPropagate()
			count += 1
			avg_err = np.average(self.total_errors)
			self.average_err[self.input_number] = avg_err
			
			if avg_err <= self.thresh:
				break
			os.system('clear')
			self.print_stats()
			print "Average of total errors: {}".format(np.average(self.total_errors))
			print("All errors: {}".format(self.total_errors))
		
		print "Ran {} times".format(count)
		self.store_weights()
	
	
	def train_mutate(self, evo_rate, runs, total_runs):
		c = 0
		self.set_inputs(0)
		
		while c < total_runs:
			for run in range(runs):
				self.feedForward()
				self.backPropagate()
			errors = np.average(self.total_errors)
			temp_weights = self.matrix_weights #copy the weights
			old_weights = self.matrix_weights
			#randomly change one of the weights in each layer.
			
			for matrix in temp_weights:
				x = np.shape(matrix)[0]
				y = np.shape(matrix)[1]
				i = rand.randint(0, x-1)
				j = rand.randint(0, y-1)
				matrix[i][j] = rand.random()*evo_rate
			
			#change weights to temp_weights and check errors
			self.matrix_weights = temp_weights
			for run in range(runs):
				self.feedForward()
				self.backPropagate()
			temp_errors = np.average(self.total_errors)
			if(temp_errors > errors):
				self.matrix_weights = old_weights
				print "Weights not changed."
			c += 1
			if np.average(self.total_errors) <= self.thresh:
				break	
			os.system('clear')
			print c
			self.print_stats()
		self.store_weights()
	
	
	def set_weight_file(self, name):
		self.weight_file_name = name	
	
	
	# WORKS. Iter and error terminated (whichever comes first): Trains inputs simultaneously.
	def train_net(self, times=1000000):
		self.input_number = 0
		self.set_inputs(0)
		c = 0
		count = 0
		#print "Training. This may take a while"
		for x in range(times):
			for i in range(self.input_length):
				self.set_inputs(i)
				self.feedForward()
				self.backPropagate()
				count += 1
				os.system("clear")
				self.print_stats()
				if(self.total_error <= self.thresh):
					print "total_error < thresh"
					continue
					if(i == self.input_length):
						break
			if(np.average(self.total_errors) <= self.thresh):
				print "Net stopped (average < thresh)"
				break
				
		if evo:
			return 1
		else:	
			self.store_weights()
			
	def focus(self, inp_number):
		pass
		
	"""			
	def train_evo(self):
		temp = self.matrix_weights
		train_net()
		#get the highest error
		for hl in range(self.)		
			hidden_errors = self.get_hidden_errors()
"""
	def change_single_weight(self, w_layer, i, j, num):
		self.matrix_weights[w_layer][i][j] = num

"""
 train_net = 0
 Train = 0 
 train_r = 0 Bad output
 train = 0
"""
