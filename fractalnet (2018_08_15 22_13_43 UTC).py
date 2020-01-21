import theano.tensor as T
from theano import shared
import theano
import numpy as np
import itertools
import typing

def flatten(mylist):
    return list(itertools.chain.from_iterable(mylist))

class net:
    weights = []
    biases = []
    consolidations = []
    weightnum = 0
    inputdims = 0
    
    def __init__(self,mylist, inputvecsize, outputvecsize, inputdimension, data, synapseThreshold):
        self.resDenominator = mylist[0]
        self.funcs = map(map(deap.compile), mylist[1:])
        self.dimensions = len(self.funcs)
        self.branchMultiplier = len(flatten(self.funcs))
        self.TBranchMult = shared(np.asscalar(np.array([self.branchMultiplier])))
        self.dataset = [shared(datum.astype(theano.config.floatX)) for (datum, outp) in data]
        self.datalabels = [shared(outp) for (datum, outp) in data]
        self.dataSample = np.array(data[0][0]) #numpy
        self.inputdimension = inputdimension
        self.genweights(inputvecsize, outputvecsize, inputdimension)
        self.threshold = synapseThreshold

    def applyFuncs(self, narray): #checked
        return [np.add.at(narray, [dim], f(narray[dim]))
                for f in self.funcs[dim] for dim in range(self.dimensions)]

    def applyFuncsMult(self,narrays):
        return flatten([self.applyFuncs(x) for x in narrays])
        
    def genConsolidate(self): #finds consolidation list from located self.dataSample
        consols = []
        
        def layerGen(arrays):
            indices = []
            for i in range(len(arrays)):
                if [x for x in flatten(indices) if x == i] == []:
                    subgroup = [i] + [j for j in range(i + 1, len(myArrays))
                                        if np.array_equal(arrays[i], arrays[j])]

                    indices.append(subgroup)
            return indices
        
        
        located = self.locate(self.dataSample)
        
        while sum(map(lambda ls: sum(map(len, ls)), consols)) < self.threshold:
            located = self.applyFuncsMult(located)
            consols.append(layerGen(located))
            located = self.positionConsolidate(located, consols[-1])
            
        #NUMPY ARRAY, NOT THEANO
        return V3IndexedShared(indices, "int32") #each sublist contains the element to which other elements should be added
                                #and the elements to add to it. If no overlap occurs, the sublist is just 1 element long
                                #before creating a sublist of an element, it checks to make sure the element is not already
                                #part of indices
    def consolidate(tensors, consolidationV2): #for theano
    	inds = consolidationV2
      for sublist in consolidationList:
          for i in sublist[1:]:  #iterate from second to last element backwards
              #increment tensor specified by sublist[0] at position [-1] by the [-1] element of the other tensors in sublist
              tensors[sublist[0]].inc_subtensor(tensors[sublist[0]][-1], tensors[sublist[i]][-1]) #use last element = activation
              del tensors[sublist[i]] #delete tensor that was added to sublist[0] tensor
      
      return tensors
        
    def positionConsolidate(narrays, consolidationList): #for numpy
        ret = []
        for sublist in consolidationList:
            ret.append(narrays[sublist[0]])
        
        return ret
        
        #this is only needed when generating consolidation arrays. Otherwise, can just use elements without spacial coordinates
        
    def locate(self, sample):  #only 1 or 2 implemented
        insize = sample.size
        sample = np.ravel(sample)
        tensorFrame = [0] * self.dimensions

        located = []
        #self.inputdimension is the dimension the input should be represented in
        if self.inputdimension == 1:
            for i in range(insize):
                myTens = tensorframe
                myTens[0] = (i * 1.0) / insize
                located.append(myTens)
            
        
        elif self.inputdimension == 2:
            if not self.inputdims:
                factors = ((i, insize/i) for i in range(int(insize**0.5),0,-1)
                            if insize % i == 0)
                self.inputdims = factors.next()
                
            x,y = self.inputdims
            for i in range(x):
                for j in range(y):
                    myTens = tensorframe
                    myTens[0] = (i * 1.0) / x
                    myTens[1] = (j * 1.0) / y
                    located.append(myTens)
        
        return located #returns list of numpy arrays in same order as sample
    
    def genweights(self, indimension, sampleinput, outsize, threshold):
        myArrays = self.locate(sampleinput)
        myWeightNum = 0
        self.hiddenlayers = 0
        while myWeightNum < threshold:
            new = self.applyFuncsMult(myArrays)
            
            weightVec = shared(np.random.rand(len(new)))
            biasVec   = shared(np.random.rand(len(new)))
            self.weights.append(weightVec)
            self.biases.append(biasVec)
            myWeightNum += len(new)
            
            self.consolidations.append(genConsolidate(new))
            myArrays = numpyConsolidate(new, self.consolidations[-1])
            
            #convert to shared data type for later speed
            self.consolidations[-1] = [shared(subArray) for subArray in self.consolidations[-1]]
            self.hiddenlayers += 1
        
        self.weights.append(shared(
            np.random.rand(len(mytensors),outsize)))        #final interconnected layer for output
        self.biases.append(shared(
            np.random.rand(len(mytensors))))
        
        myWeightNum += len(mytensors) * outsize
        self.weightNum = myWeightNum
            
    def feedForward(self, inp, weights, biases, consolidations, hiddenlayers):
        inputs = T.ravel(inp)
        
        for i in range(hiddenlayers):
            new =
            new = new * weights[i] + biases[i]
            new = self.consolidate(new, consolidations[i])
            new = [T.tanh(tensor) for tensor in new]
            inputs = new
        
        inputs = theano.dot(inputs, self.weights[-1])
        inputs += self.biases[-1]
        return T.tanh(inputs)
        
    def error(selif,inp, outp):
        diff = outp - self.feedForward(inp)
        return T.sum(T.dot(diff, diff))
        
    def train(self, num_epochs):

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
            
        len(logbook.select('gen'))