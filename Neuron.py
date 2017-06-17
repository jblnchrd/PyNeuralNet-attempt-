import numpy as np
import random as rand
path = '/home/jsurg/Programs/python/NeuralNet/resource/'
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
		self.gradient = 0.
		self.next_nodes = None
		self.from_nodes = None
		self.testing = testing
		self.out_error = 0.0
		#initialize weights and attach connections to each neuron for each output.
		#weights are in the neuron in the NEXT LAYER. i.e. From first hidden layer to last layer.
		
		if firstLayer is False:
			self.layer_num = layerNumber
			for inp in xrange(self.inputs): #inputs to this neuron.
				r = 2*rand.random() - 1
				self.weights.append(r)
				
				
		if val is not None:
			self.value = val
				
	def getOutputError(self):
		return self.out_error
		
	
	def setOutputError(self, err):
		self.out_error = float(err)
	
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


