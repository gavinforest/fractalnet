from random import shuffle
import numpy as np
import numpy.linalg as LA
import theano
import theano.tensor as T
import itertools
import time
import math
from itertools import izip
from theano import shared,config,function
from collections import OrderedDict
theano.config.mode = 'FAST_RUN'



SPACESPAN = 2.0

def box(x):
    if x > 0 + SPACESPAN / 2.0:
        return -1.0 + x
    elif x < 0 - SPACESPAN / 2.0:
        return 1.0 + x
    else:
        return x
            
def discretize(x, resDenominator):
    if x > 0.0:
        return np.floor(1.0 * x * resDenominator) / resDenominator
    elif x < 0.0:
        return np.floor(1.0 * x * resDenominator) / resDenominator
    else:
        return 0.0


def flatten(mylist):
    return list(itertools.chain.from_iterable(mylist))

def singleGenConsolidate(arrays):
    indices = []
    flattenedIndices = []
    arraylen = len(arrays)
    initArraylen = len(arrays) * 1.0
    irange = range(arraylen)
    
    counter = True
    print "starting a singleGenConsolidate"
    for i in irange:
        if i not in flattenedIndices:
            subgroup = [i] + [j for j in irange[i + 1:] if np.array_equal(arrays[i], arrays[j])]
            arraylen -= len(subgroup)
            indices.append(subgroup)
            for x in subgroup:
                flattenedIndices.append(x)
            if counter and arraylen/initArraylen < 0.5:
                print "half way done a singleGenConsolidate"
                counter = False
    
    print "finished a singleGenConsolidate"
    return indices

def positionConsolidate(arrays, consolidations):
    return [arrays[sublist[0]] for sublist in consolidations]

def numpyConsolidate(array, consols):
    new = []
    for sublist in consols:
        x = 0.0
        for i in sublist:
            x += array[i]
        new.append(x)

    return new

def generalfeedforward((weights, biases, finalweights, finalbiases), consolidations, hiddenlayers, branchMultiplier, inp):
    for i in range(hiddenlayers):
        inp = np.repeat(inp, branchMultiplier)
        inp = np.multiply(weights[i], inp)
        inp = numpyConsolidate(inp, consolidations[i])
        inp = np.tanh(np.add(inp, biases[i]))
    
    print inp
    finalout = np.tanh(np.dot(inp, finalweights) + finalbiases)
    return finalout

class nnet:
    weights = []
    biases = []
    consolidations = []
    finalweights = []
    finalbiases = []
    weightnum = 0
    hiddenlayers = 0
    
    def __init__(self, resolution, functions, inputdimension, datainp, dataoutp, synapseThreshold):
        self.resDenominator = resolution
        self.funcs = functions
        self.dimensions = len(self.funcs)
        self.branchMultiplier = len(flatten(self.funcs))
        self.TBranchMult = shared(np.array([self.branchMultiplier]))
#         self.dataset = [shared(dat.astype(theano.config.floatX)) for dat in datainp]
#         self.datalabels = [shared(outp.astype(theano.config.floatX)) for outp in dataoutp]
        self.dataSample = datainp[0] #numpy
        self.dataOutSample = dataoutp[0]
        self.inputdimension = inputdimension
        
        if self.dataSample.size ** (1.0/ inputdimension) > self.resDenominator * 2.0:
            print "resDenominator may be too small for effective learning"
            
        self.threshold = synapseThreshold

    def applyFuncs(self, narray): #checked
        boxvec = np.vectorize(lambda x: discretize(box(x), self.resDenominator))
        displaced = []
        
        for dim in range(self.dimensions):
            for f in self.funcs[dim]:
                
                zeroes = np.zeros_like(narray)
                displacement = f(narray[dim]) % SPACESPAN
                zeroes[dim] = displacement
                
                newArray = np.add(narray, zeroes)
                newArrayBoxed = boxvec(newArray)
                displaced.append(newArrayBoxed)
                
        
        return displaced
        

    def applyFuncsMult(self, narrays): 
#         print "narrays: " + str(narrays)
        return flatten([self.applyFuncs(x) for x in narrays])
    
    def locate(self, sample):  #only 1 or 2 implemented
        insize = sample.size
        sample = np.ravel(sample) 
        tensorFrame = [0] * self.dimensions
        located = [0] * insize
        
        boxvec = np.vectorize(lambda x: discretize(box(x), self.resDenominator))
        
        #self.inputdimension is the dimension the input should be represented in
        
        if self.inputdimension == 1:
            
            for i in range(insize):
                myTens = tensorFrame[:]
                myTens[0] = (2.0 * i) / insize - 1.0
                located[i] = boxvec(np.array(myTens))
        
        if self.inputdimension == 2:
            located = []
            factorPairs = [(i,(insize / i)) for i in range(1, int(math.floor(insize**0.5))) if insize % i == 0]
            pair = factorPairs[-1]
            for i in range(pair[0]):
                for j in range(pair[1]):
                    myTens = tensorFrame[:]
                    myTens[0] = 2.0 * i/pair[0] - 1.0
                    myTens[1] = 2.0 * j/pair[1] - 1.0
                    located.append(boxvec(np.array(myTens)))
                            
        return located
        
    def genConsolidate(self): #finds consolidation list from located self.dataSample
        consols = []
#         print len(self.dataSample.ravel())
        located = self.locate(self.dataSample)
#         print len(located)
#         print type(located[0])
        print "running initial singleGenConsolidate"
        located = self.applyFuncsMult(located)
        consols.append(singleGenConsolidate(located))
        located = positionConsolidate(located, consols[-1]) 

#         print located
        print "starting to loop"
        while len(flatten(flatten(consols))) + len(flatten(consols[-1])) * self.branchMultiplier < self.threshold:
            located = self.applyFuncsMult(located)
            consols.append(singleGenConsolidate(located))
            located = positionConsolidate(located, consols[-1]) 
            print "looped"
            
        self.consolidations = consols
        return consols
    
    def genWeights(self):
        myWeightNum = 0
        self.hiddenlayers = 0
        self.weights = []
        self.biases = []
        self.finalweights = []
        self.finalbiases = []
        for consol in self.consolidations:
            weightVec = (np.random.rand(len(flatten(consol))) * 0.05).astype(theano.config.floatX).tolist()
            biasVec   = (np.random.rand(len(consol)) * 0.1).astype(theano.config.floatX).tolist()
            self.weights.append(weightVec)
            self.biases.append(biasVec)
            myWeightNum += len(weightVec)

            #convert to shared data type for later speed
            self.hiddenlayers += 1
        
        outsize = len(self.dataOutSample)
        
        self.finalweights = np.random.rand(len(self.consolidations[-1]),outsize)        #final interconnected layer for output
        self.finalbiases = np.random.rand(outsize)

        myWeightNum += len(self.consolidations[-1]) * outsize
        self.weightNum = myWeightNum
        
    def feedforward(self, inp):
        def foo(x):
            if x > 0.0:
                return x
            else:
                return 0.0
        relu = np.vectorize(foo)
        for i in range(self.hiddenlayers):
            inp = np.repeat(inp, self.branchMultiplier)
#             print "repeated: " + str(inp)
            inp = np.multiply(self.weights[i], inp)
#             print "weighted: " + str(inp)
            inp = numpyConsolidate(inp, self.consolidations[i])
#             print "consolidated: " + str(inp)
            inp = relu(np.add(inp, self.biases[i]))
            print "biased and tanned: " + str(inp)
        
        print "before finals: " + str(inp)
        finalout = relu(np.dot(inp, self.finalweights) + self.finalbiases)
        return finalout
    
    def test(inputs, outputs):
        def foo(x):
            if x > 0.5:
                return 1.0
            return 0.0

        binarize = np.vectorize(foo)
        
        errors = [LA.norm(outp - binarize(self.feedforward(inp))) for inp,outp in izip(inputs,outputs)]
        return sum(errors)/len(inputs) * 100.0

# singleGenConsolidate([np.array([0,0,0]), np.array([0,1,2,3]), np.array([0,0,0])])
# testnet = nnet(10, [[lambda x: x + 1], [lambda x: x * 2]], 1, [np.array([0,1,2,3])], [np.array([1])], 100)
# testnet.genConsolidate()
# testnet.genWeights()
# numpyConsolidate(np.array([1,2,3]), [[0,2], [1]])
# testnet.feedforward(np.array([0,1,2,3]))


flatten = lambda mylist: list(itertools.chain.from_iterable(mylist))

class V2IndexedShared:
    def __init__(self, atype):
        self.myType = atype
    
    def fromList(self,D2List):
        self.complete = shared(np.array(flatten(D2List)).astype(self.myType))
        self.length = shared(len(D2List))
        inds = [0]
        for sublist in D2List:
            inds.append(len(sublist) + inds[-1])
        self.inds = shared(np.array(inds).astype('int32'))
        return self

    
class V3IndexedShared:
    def __init__(self,D3List, atype):
        self.complete = shared(np.array(flatten(flatten(D3List))).astype(atype))
#         self.complete = np.array(flatten(flatten(D3List))).astype(atype)

        self.myType = atype
        self.length = shared(len(D3List))
        D2Inds = [0]
        D3Inds = [0]
        
        for sublistOLists in D3List:
            
            for sublist in sublistOLists:
                D2Inds.append(len(sublist) + D2Inds[-1])
                
            D3Inds.append(len(sublistOLists) + D3Inds[-1])
            
                
        self.D2Inds = shared(np.array(D2Inds).astype('int32'))
        self.D3Inds = shared(np.array(D3Inds).astype('int32'))
        
#         self.D2Inds = np.array(D2Inds).astype('int32')
#         self.D3Inds = np.array(D3Inds).astype('int32')
#         print D3Inds
#         print D2Inds
#         print np.array(flatten(flatten(D3List)))
    

relu = theano.tensor.nnet.relu


def getSubV2(index, inds, complete):
    return complete[inds[index]:inds[index + 1]]

def getSubV2Check():
    tind = T.iscalar('ind')
    tinds = T.ivector('inds')
    tcomp = T.vector('complete')
    
    nind = np.asscalar(np.array([0]).astype('int32'))
    ninds = np.array([0,2,4]).astype('int32')
    comp = np.array([0,1,2,3,4]).astype(theano.config.floatX)
    
    f = theano.function([tind, tinds, tcomp], getSubV2(tind, tinds, tcomp))
    print f(nind,ninds,comp)


def getSubV3(index, indinds, inds, complete):
    subInds = inds[indinds[index] : indinds[index + 1] + 1]
    return subInds, complete

def getSubV3Check():
    D3List = [[[0,1,2,], [3,4,5]], [[6,7,8], [9,10,11]]]
    consolcomplete = [0,1,2,3,4,5,6,7,8,9,10,11]
    consolinds = [0,3,6,9,12]
    consolindinds = [0,2,4]
    print getSubV3(1, consolindinds, consolinds, consolcomplete)
    print consolinds[2:5]
    t = V3IndexedShared(D3List, 'int32')
    
# getSubV3Check()
    
def singleConsol(tensor, indices):
    return (T.choose(indices, tensor)).sum()

def consolidateTensor(tensor, consolinds, consolcomplete):
    irange = T.arange(consolinds.size - 1)
    consolidated, updates = theano.scan(fn = lambda ind, tens, coninds, concomp: singleConsol(tens, getSubV2(ind, coninds, concomp)),
                                        sequences = irange,
                                        non_sequences=[tensor, consolinds, consolcomplete])
    return consolidated

def consolidateTensorCheck():
    x = T.vector('x')
    inds = T.ivector('inds')
    complete = T.ivector('complete')
    y = consolidateTensor(x,inds,complete)
    f = theano.function([x,inds,complete], y)
    print f(np.array([1.0,2.0,4.0,8.0]).astype(theano.config.floatX), np.array([0,3,4]).astype('int32'), np.array([1,2,3,0]).astype('int32'))



class tnnet:
    def __init__(self, resolution, functions, inputdimension, traindatainps, traindataoutps, testdatainps, testdataoutps, synapseThreshold, net = None):
        
        mynet = None
        if net == None:
            mynet = nnet(resolution, functions, inputdimension, traindatainps, traindataoutps, synapseThreshold)
            mynet.genConsolidate()
            mynet.genWeights()
        else:
            mynet = net
        self.nnet = mynet

        self.weights = map(lambda x: shared(np.array(x).astype(theano.config.floatX)), mynet.weights)

        self.biases = map(lambda x: shared(np.array(x).astype(theano.config.floatX)), mynet.biases)
    
        self.fws = shared((mynet.finalweights).astype(theano.config.floatX))
        self.fbs = shared((mynet.finalbiases).astype(theano.config.floatX))
        self.bMult = shared(np.asscalar(np.array([mynet.branchMultiplier]).astype('int32')))
        self.weightnum = mynet.weightnum
        self.hiddenlayers = mynet.hiddenlayers

        self.sharedTrainingInps = shared(np.array(traindatainps).astype(theano.config.floatX))
        self.sharedTrainingOutps = shared(np.array(traindataoutps).astype(theano.config.floatX))
        self.sharedTestingInps = shared(np.array(testdatainps).astype(theano.config.floatX))
        self.sharedTestingOutps = shared(np.array(testdataoutps).astype(theano.config.floatX))

        self.consolMasks = []
        for layerConsols in mynet.consolidations:
            print layerConsols
            inLength = len(flatten(layerConsols))
            outLength = len(layerConsols)
            frame = [[0.0] * inLength] * outLength
            for i in range(outLength):
                subConsol = layerConsols[i]
                for ind in subConsol:
                    frame[i][ind] = 1.0
            
            self.consolMasks.append(shared(np.array(frame).astype(theano.config.floatX)))

#         self.consolMasks = []
#         for layerConsols in mynet.consolidations:
# #             print "initial " + str(layerConsols)
#             maxInLength = max(map(len, layerConsols))
#             inLength = len(flatten(layerConsols))
#             extended = map(lambda l: l + [inLength] * (maxInLength - len(l)), layerConsols)
# #             print "extended: " + str(extended)
#             self.consolMasks.append(shared(np.array(extended).astype('int32')))
        
        print "mask sizes " + str( [cons.get_value().size for cons in self.consolMasks])
        inps = T.fmatrix('inps')
        actualOutps = T.fmatrix('actualOutps')
        alpha = T.fscalar('alpha')
        index = T.iscalar('index')
        batchSize = T.iscalar('batchSize')
        

        self.params = self.weights + self.biases + [self.fws, self.fbs]
        def layer(ind, x):

            x = T.repeat(x, self.bMult, axis = 0) * self.weights[ind]
            x = T.dot(self.consolMasks[ind], x)
            # x = T.concatenate([x, T.zeros_like(x)])
            # x = x.take(self.consolMasks[ind]).sum(axis = 1)

            return theano.tensor.nnet.relu(x + self.biases[ind])

        final = T.fmatrix()
        def loop(z):
            ret = T.fmatrix()
            for i in range(self.hiddenlayers):
                z = layer(i, z)
                ret = z
            return ret
        final1 = loop(inps)
        final = T.dot(final1, self.fws)
        final = final + self.fbs
        final = theano.tensor.nnet.softmax(final)
        
        self.classify = theano.function([inps], final)
        
        error = T.mean(T.nnet.binary_crossentropy(final, actualOutps))
        
        gradients = T.grad(error, self.params)
        param_Updates = OrderedDict((p, p - alpha * g) for p, g in zip(self.params, gradients))
        self.train = theano.function(inputs = [index, alpha, batchSize],
                                                                    outputs = error,
                                                                    updates = param_Updates,
                                                                    givens = {
                                                                        inps : self.sharedTrainingInps[index:index + batchSize],
                                                                        actualOutps: self.sharedTrainingOutps[index: index + batchSize]
                                                                    })
        self.test = theano.function(inputs = [index],
            outputs = final,
            givens = {
                inps : self.sharedTestingInps[index]
            })
                                                                    
    
    def descend(self, alpha, epochs, trainingInps, trainingOutps, testingInps, testingOutps, verbose):
        training = zip(trainingInps, trainingOutps)
        testing = zip(testingInps, testingOutps)
        testLen = len(testing)
        batchSize = 15
        testingAccuracies = []
        
        for i in range(epochs):
            print "starting epoch..."
            start = time.time()
            for j in range(math.floor(len(trainingInps))/(batchSize * 1.0)):
                self.train(np.asscalar(np.array([j]).astype('int32')), alpha)
            
            testingAccuracy = 0.0
            for j in range(len(testingOutps)):
                if np.argmax(self.test(np.asscalar(np.array([j]).astype('int32')))) == np.argmax(testingOutps[j]):
                    testingAccuracy += 1.0
            # for (testInp, testOutp) in testing:
            #     if np.argmax(self.classify(testInp)) == np.argmax(testOutp):
            #         testingAccuracy += 1.0
                    
            percentTestingAccuracy = testingAccuracy / testLen * 100.0
            testingAccuracies.append(percentTestingAccuracy)
            end = time.time()
            if verbose:
                print "epoch " + str(i + 1) + " -- testing accuracy : " + str(percentTestingAccuracy) + " duration: " + str(end - start) + "s"
        
        return max(testingAccuracies), testingAccuracies