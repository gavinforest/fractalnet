import theano.tensor as T
from theano import shared
import theano
import numpy as np
import itertools

def flatten(mylist):
	return list(itertools.chain.from_iterable(mylist))

class net:
	weights = []
	biases = []
	consolidations = []
	weightnum = 0
	inputdims = 0
	
	def __init__(self,mylist, inputvecsize, outputvecsize, inputdimension, data):
		self.resDenominator = mylist[0]
		self.funcs = mylist[1:]
		self.dimensions = len(self.funcs)
		self.branchMultiplier = len(flatten(self.funcs))
		self.dataset = [shared(datum.astype(theano.config.floatX)) for (datum, outp) in data]
		self.datalabels = [shared(outp) for (datum, outp) in data]
		self.genweights(inputvecsize, outputvecsize, inputdimension)
	
	def applyFuncs(tensor, funcs):
		ret = []
		for i in range(self.dimensions):
			for f in funcs[i]:
				ret.append(tensor.inc_subtensor(tensor[i], f(tensor[i])))
		return ret
		
	def applyFuncsMult(tensors, funcs):
		return flatten([applyFuncs(x) for x in tensors])
		
	def genConsolidate(self, mytensors): #takes a list of tensors and finds consolidation list
		indices = []
		for i in range(len(mytensors)):
			
			subgroup = [i]
			for j in range(i,len(mytensors)):
				if tensor[i] == tensor[j]:
					subgroup.append(j)
			
			indices.append(shared(np.array(subgroup)))
			for index in subgroup[::-1]:	#iterate backwards so deletion does not change previous indices
				del tensors[index]
		
		return indices #each sublist contains the element to which other elements should be added
						#and the elements to add to it. They are then deleted and iteration continues
		
	def consolidate(tensors, consolidationList): 
		for sublist in consolidationList:
			for i in range(T.prod(sublist.shape),1,-1):  #iterate from second to last element backwards
				tensors[sublist[0]].inc_subtensor(tensors[sublist[0]][-1], tensors[sublist[i]][-1])
				del tensors[index]
		
		return tensors
		
	def locate(self, sample, inputdimension):  #only 1 or 2 implemented
		insize = sample.size
		sample = np.ravel(sample)
		tensorFrame = [0] * (self.dimensions + 1)
		located = []
		
		if inputdimension == 1:
			self.inputdimension = 1
			indexed = []
			for i in range(insize):
				myTens = tensorframe
				myTens[0] = i
				myTens[-1] = sample[i]
				indexed.append(myTens)
			located = [shared(thing) for thing in indexed]
			return located
		
		elif inputdimension == 2:
			self.inputdimension = 2
			indexed = []
			if not self.inputdims:
				for i in range(insize**0.5,1, -1):
					if insize % n == 0:
						self.inputdims = (i, insize / i)
			x,y = self.inputdims
			for i in range(x):
				for j in range(y):
					myTens = tensorframe
					myTens[0] = i
					myTens[1] = j
					myTens[-1] = sample[i * x + j]
					indexed.append(myTens)
			located = [shared(thing) for thing in indexed]
			return located
			
		
		
	
	def genweights(self, indimension, sampleinput, outsize, threshold):
		mytensors = self.locate(sampleinput, inputdimension)
		myWeightNum = 0
		self.hiddenlayers = 0
		while myWeightNum < threshold:
			new = applyFuncsMult(mytensors, self.funcs)
			
			weightVec = shared(np.random.rand(len(new)))
			biasVec   = shared(np.random.rand(len(new)))
			self.weights.append(weightVec)
			self.biases.append(biasVec)
			myWeightNum += len(new)
			
			self.consolidations.append(genConsolidate(new))
			new = consolidate(new, self.consolidations[-1])
			mytensors = new
			self.hiddenlayers += 1
		
		self.weights.append(shared(
			np.randon.rand(len(mytensors),outsize)))		#final interconnected layer for output
		self.biases.append(shared(
			np.random.rand(len(mytensors))))
		
		myWeightNum += len(mytensors) * outsize
		self.weightNum = myWeightNum
			
			
	def feedForward(self, inp):
		inputs = self.locate(inp, self.inputdimension)
		
		for i in range(self.hiddenlayers):
			new = self.applyFuncsMult(inputs, self.funcs)
			new = [tensor.set_subtensor(tensor[-1], T.tanh(tensor[-1] * w + b)) 
					for tensor, w, b in zip(new, self.weights[i], self.biases[i])]
			new = self.consolidate(new, self.consolidations[i])
			inputs = new
		
		inputs = shared(np.array([tensor[-1] for tensor in inputs]))
		
		inputs = theano.dot(inputs, self.weights[-1])
		
		inputs += self.biases[-1]
		
		return T.tanh(inputs)
		
	def error(self,inp, outp):
		prediction = self.feedForward(inp)
		diff = outp - prediction
		return T.sum(T.dot(diff, diff))
		
	def train(self, num_epochs):
		prevError = 1
		currError = 0.1
		
		ind = T.iscalar('index')
		x = T.tensor('input')
		y = T.tensor('output')
		err = T.scalar(self.error(x,y))
		error = theano.function([x,y], err)
		
		gw = [theano.grad(error,weight) for weight in self.weights]
		gb = [theano.grad(error,bias)   for bias in self.biases]
		updates = [(weight, weight - wgrad * alpha) for weight, wgrad in zip(self.weights, gw)] +\
				  [(bias  , bias  -  bgrad * alpha) for bias ,  bgrad in zip(self.biases , gb)]
		train = theano.function(
								inputs = [ind],
								outputs = err,
								updates = updates,
								given = {
									x : self.dataset[ind],
									y : self.datalabels[ind] }
								)
		print '... training'
		for i in range(num_epochs):
			errors = []
			for ind in range(len(self.datalabels)):
				errors.append(train(ind))
			print "epoch " + str(i) + " complete, error is " + sum(errors)/len(self.datalabels)
			
		
		
			
		
		
		
