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
#imports
import numpy as np
import random as rand
import time
from math import *
from tempfile import TemporaryFile
import os
import time 
from Net import *


"""
Credit to Dave Miller for some ideas about initializing the network using topology. He has a great video designing a simple
feedforward neural net in c++ at https://www.youtube.com/watch?v=KkwX7FkLfug

There is a weight matrix for each connection between layers, with the number of rows being equal 
to the number of neurons in the layer holding these weights. The columns equal the number of 
neurons in the previous layer.
To feedForward the network, we simply multiply the input matrix (column vector) by the weight matrix.
We get the values for each neuron based on the output vector. These are attached to the hidden layer
neurons and used for the inputs for the next calculation.

Right now there are two ways to use the network. Either using matrices without storing any values
inside the Neuron objects (yet), or using the neurons without matrices, which seems a bit too complicated than it needs to be.
Just use the matrices, it's way easier.

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

Update May 16, 2017: The matrix form of the feedforward member function of the Net class needs to store the values
of the first calculation (first weight matrix times input matrix) in the hidden layer, but first we can
simply bypass storing them in the neuron objects while we get the calculation right. These values must
be set so that we can use these values to back propagate the network.

Update May 25, 2017.
Back propagation will be primarily done through objects, that is, node objects with the incoming weights
stored in the neurons themselves. Calculate all the errors first, then use these values to get the
new weights for the network.

Update June 7, 2017:
Storage of weigts using np.load and np.save is working correctly and should work regardless of platform. Next 
we need to split up the classes into their own files, and perhaps make sure the non-matrix form works properly.
"""
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
	with open(file_name, 'a+') as f:
		f.write(weights)


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
	print("Welcome to my custom neural network!")
	top = [10, 12, 10, 1] # topology specifies the number of layers and nodes in each layer.
	#specify inputs and outputs. Then the topology can be used
	
	# For example, [2, 1, 2] means 2 inputs, 1 hidden node, and 2 outputs.
	inps = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 1])
	targets = [1.]
	#Net(top, inps, targs, matrix=True, load_=True, weightFile=fname, test=True)
	#myNet = Net(top, inps, targets, matrix=True, load_=False, weightFile="weights", test=False)	
	#myNet.setTargetValues(targets)
	#times = input("Enter number of times to train\n:")
	#times = 10000
	#c = myNet.train(times)
	#print "Ran {} times".format(c)
	
	myNet_trained = Net(top, inps, targets, matrix=True, load_=True, weightFile="weights", test=False)
	myNet_trained.ffMatrix()
	myNet_trained.print_outputs()
	#record_weights(myNet_trained.get_weights(), "weightstxt.txt")
	print "Neural Net finished successfully..."
