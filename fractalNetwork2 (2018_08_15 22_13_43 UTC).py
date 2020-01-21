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
import multiprocessing



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

def getSubgroup(ind):
    return [ind] + [j for j in range(ind + 1, len(globalSingleGenArrays)) if np.array_equal(globalSingleGenArrays[ind], globalSingleGenArrays[j])]

WORKERS = 4
globalSingleGenArrays = None

def singleGenConsolidate(arrays):
    global globalSingleGenArrays
    globalSingleGenArrays = arrays
    indices = []
    flattenedIndices = []
    arraylen = len(arrays)
    initArraylen = len(arrays) * 1.0
    irange = range(arraylen)


    p = multiprocessing.Pool(WORKERS)
    flag = True
    print "starting a singleGenConsolidate"
    for i in irange:
        if i not in flattenedIndices:
            
            nextCouple = [i]
            counter = i + 1
            while len(nextCouple) < WORKERS and counter < arraylen:
                if [k for k in nextCouple if np.array_equal(arrays[k], arrays[counter])] == [] and (counter not in flattenedIndices):
                    nextCouple.append(counter)
                counter +=1

            subgroups = p.map(getSubgroup, nextCouple)
            for subgroup in subgroups:
                arraylen -= len(subgroup)
                indices.append(subgroup)
                flattenedIndices = flattenedIndices + subgroup
                if flag and arraylen/initArraylen < 0.5:
                    print "half way done a singleGenConsolidate"
                    flag = False
    p.close()
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
    
    def __init__(self, resolution, functions, inputdimension, datainp, dataoutp, synapseThreshold, regression):
        self.resDenominator = resolution
        self.funcs = functions
        self.dimensions = len(self.funcs)
        self.branchMultiplier = len(flatten(self.funcs))
        # self.TBranchMult = shared(np.array([self.branchMultiplier]))
#         self.dataset = [shared(dat.astype(theano.config.floatX)) for dat in datainp]
#         self.datalabels = [shared(outp.astype(theano.config.floatX)) for outp in dataoutp]
        self.dataSample = datainp[0] #numpy
        self.dataOutSample = dataoutp[0]
        self.inputdimension = inputdimension
        self.regression = regression
        
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
            weightVec = None
            biasVec = None

            if self.regression:
                weightVec = (np.random.rand(len(flatten(consol))) * 0.04).astype(theano.config.floatX).tolist()
                biasVec   = (np.random.rand(len(consol)) * 0.1).astype(theano.config.floatX).tolist()
            else:
                weightVec = (np.random.randn(len(flatten(consol))) * 0.05).astype(theano.config.floatX).tolist()
                biasVec   = (np.random.rand(len(consol)) * 0.1).astype(theano.config.floatX).tolist()

            self.weights.append(weightVec)
            self.biases.append(biasVec)
            myWeightNum += len(weightVec)

            #convert to shared data type for later speed
            self.hiddenlayers += 1
        
        outsize = len(self.dataOutSample)
        
        self.finalweights = np.random.rand(len(self.consolidations[-1]),outsize) * 0.04       #final interconnected layer for output
        self.finalbiases = np.random.rand(outsize) * 0.1

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
    def __init__(self, resolution, functions, weightSharing, inputdimension, traindatainps, traindataoutps, testdatainps, testdataoutps, synapseThreshold, regression = False, net = None):
        
        mynet = None
        if net == None:
            mynet = nnet(resolution, functions, inputdimension, traindatainps, traindataoutps, synapseThreshold, regression)
            mynet.genConsolidate()
            mynet.genWeights()
        else:
            mynet = net
        self.nnet = mynet
        
        self.regression = regression
        
        self.sharedTrainingInps = shared(np.array(traindatainps).astype(theano.config.floatX))
        self.sharedTrainingOutps = shared(np.array(traindataoutps).astype(theano.config.floatX))
        self.sharedTestingInps = shared(np.array(testdatainps).astype(theano.config.floatX))
        self.sharedTestingOutps = shared(np.array(testdataoutps).astype(theano.config.floatX))


#         print "weights: " + str(mynet.weights)
#         print "biases: " + str(mynet.biases)
#         print mynet.feedforward(np.array([0,1,2,3]).astype(theano.config.floatX))
#         inp_lengths = [len(consol) for consol in mynet.consolidations]
#         inp_lengths = [datainps[0].size] + inp_lengths
#         self.inp_lengths = np.array(inp_lengths).astype('int32')
#         print "inp_lengths: " + str(inp_lengths)
#         print "dinp length: " + str(datainps[0].size)
#         print "consolidations: " + str(mynet.consolidations)
        
#         self.maxInp = np.amax(inp_lengths)
#         self.maxPad = shared(np.array([0] * (np.amax(inp_lengths) - min(inp_lengths))).astype(theano.config.floatX))
#         self.maxZeroes = shared(np.array([0.0] * self.maxInp).astype(theano.config.floatX))
        
#         initialPaddingLength = np.amax(inp_lengths) - datainps[0].size
#         self.initialPadding = shared(np.array([0.0] * initialPaddingLength).astype(theano.config.floatX))
#         self.inp_lengths = shared(self.inp_lengths)
        self.weights = mynet.weights
        self.weightSharing = weightSharing
        self.weightSharing += [False] * (len(self.weights) - len(self.weightSharing))
        for i in range(len(self.weights)):
          if self.weightSharing[i]:
            self.weights[i] = self.weights[i][:mynet.branchMultiplier]
        self.weights = map(lambda x: shared(np.array(x).astype(theano.config.floatX)), self.weights)
        
#         print tbiases.complete.get_value()
        self.biases = map(lambda x: shared(np.array(x).astype(theano.config.floatX)), mynet.biases)
    
        tcons = V3IndexedShared(mynet.consolidations, 'int32')
#         print "final weights: " + str(mynet.finalweights)
        tfws = shared((mynet.finalweights).astype(theano.config.floatX))
        tfbs = shared((mynet.finalbiases).astype(theano.config.floatX))
        self.weightnum = mynet.weightnum
        self.hiddenlayers = mynet.hiddenlayers
        
        self.consolMasks = []
        self.dEdWsinverseConsols = []
        self.oneMultipliers = []
        for layerConsols in mynet.consolidations:
#             print "initial " + str(layerConsols)
            maxInLength = max(map(len, layerConsols))
            inLength = len(flatten(layerConsols))
            extended = map(lambda l: l + [inLength] * (maxInLength - len(l)), layerConsols)
#             print "extended: " + str(extended)
            self.consolMasks.append(shared(np.array(extended).astype('int32')))
            self.oneMultipliers.append(shared(np.array([1.0] * maxInLength).astype(theano.config.floatX)))

            
            flattened = flatten(layerConsols)
            frame = [0] * len(flattened)
            for i in range(len(layerConsols)):
                for sub in layerConsols[i]:
                    frame[sub] = i
            
            self.dEdWsinverseConsols.append(shared(np.array(frame).astype('int32')))
        
        # print self.nnet.consolidations[0]
        # print self.dEdWsinverseConsols[0].get_value()
        
        softmaxInp = T.fvector()
        softmaxOut = T.nnet.softmax(softmaxInp)
        
        
        inp = T.fvector('inp')
#         paddedInp = T.concatenate([inp, self.initialPadding])
        actualOutp = T.fvector('actualOutp')
        alpha = T.fscalar('alpha')
        
        self.consolidations = map(lambda x: V2IndexedShared('int32').fromList(x), self.nnet.consolidations)
        
        self.consindinds = tcons.D3Inds
        self.consinds = tcons.D2Inds
        self.cons = tcons.complete
        self.bMult = shared(np.asscalar(np.array([mynet.branchMultiplier]).astype('int32')))
        self.fws = tfws
        self.fbs = tfbs
        
        self.zeropad = shared(np.array([0.0]).astype(theano.config.floatX))
        
        self.params = self.weights + self.biases + [self.fws, self.fbs]
#         self.constparams = [self.consindinds, self.consinds, self.cons, self.winds, self.binds, self.bMult]
        
#         def singleCons(consInds, tens):
#             asum, _ = theano.scan(fn = lambda i, tensor: tensor[i],
#                                  sequences = consInds,
#                                  non_sequences = tens)
#             return asum.sum()
        
        
#         def layer(ind, inpLength, inp):
#             weights = getSubV2(ind, self.winds, self.ws)
#             biases = getSubV2(ind, self.binds, self.bs)
#             consolinds, consols = getSubV3(ind, self.consindinds, self.consinds, self.cons)
#             inp = T.repeat(inp[0:inpLength], self.bMult)
#             inp = weights * inp
            
#             #consolidation code
#             numConsolidatedRange = T.arange(consolinds.size - 1)
#             inp, _ = theano.scan(fn = lambda ind, x, coninds, concomp: singleCons(getSubV2(ind, coninds, concomp), x),
#                                                 sequences = numConsolidatedRange,
#                                                 non_sequences=[inp, consolinds, consols])
#             inp = inp + biases
#             inp = relu(inp)
#             padding = self.maxPad[0 : self.maxInp - inp.size]
#             inp = T.concatenate([inp, padding])
#             return inp


        def gradLayer(ind, nextdEdInputs, layerOutput): #returns (dEdWs, dEdBs, newdEdInputs)
            expandedLayerOutput = T.repeat(layerOutput, self.bMult)
            inverse = self.dEdWsinverseConsols[ind]
            inverseddEdInputs = nextdEdInputs.take(inverse)
            dEdWs = expandedLayerOutput * inverseddEdInputs

            if self.weightSharing[ind]:
              dEdWs = dEdWs.reshape((dEdWs.size // self.bMult, self.bMult)).sum(axis = 0)
            dEdBs = nextdEdInputs
            #upt to here correct
            
            dOutsdIns = T.fvector()
            if self.regression:
                dOutsdIns = T.gt(layerOutput, T.zeros_like(layerOutput))
                dOutsdIns = T.cast(dOutsdIns, 'float32')
            else:
                dOutsdIns = T.ones_like(layerOutput) - (layerOutput * layerOutput)

            dEdConnections = T.fvector()
            if self.weightSharing[ind]:
                dEdConnections = T.tile(self.weights[ind], layerOutput.size) * inverseddEdInputs
            else:
                dEdConnections = self.weights[ind] * inverseddEdInputs

            dEdConnectionGroups = dEdConnections.reshape((dEdConnections.size // self.bMult, self.bMult))
            dEdOuts = T.dot(dEdConnectionGroups, T.ones((self.bMult), 'float32')) #could need to be axis 1
            
            newdEdInputs = dEdOuts * dOutsdIns
            
            return (dEdWs, dEdBs, newdEdInputs)
        
        def intermediateLoop(z):
            ret = []
            for i in range(self.hiddenlayers):
                z = layer(i, z)
                ret.append(z.copy())
            
            return ret
        
        def myGrad(theta, actualOutp):
            outputs = intermediateLoop(theta)
            outputs = [theta] + outputs
            last = T.dot(outputs[-1], self.fws)
            last = last + self.fbs
            
            finaldOutdIn = T.fvector()
            err = T.fvector()
            initdEdIn = T.fvector()
            predfinal = T.fvector()
            
            if regression:
                predfinal = theano.tensor.nnet.relu(last)
                
                difference = predfinal -actualOutp
                err = T.dot(difference, difference)
                
                finaldOutdIn = T.gt(predfinal, T.zeros_like(predfinal))
                finaldOutdIn = T.cast(finaldOutdIn, 'float32')
                initdEdIn = finaldOutdIn * difference
                
            else:
                predfinal = T.nnet.softmax(last)[0]
                initdEdIn = predfinal - actualOutp
                

            
            
            dEdWs = []
            dEdBs = []
            
#             difference = predfinal -actualOutp
#             err = T.dot(difference, difference)
            

            dEdFbs = initdEdIn
            
            
            initdEdInshuffled = initdEdIn.dimshuffle(('x',0))
            finalStructuredLayerOut = outputs[-1]
            finalStructuredLayerOutShuffled = finalStructuredLayerOut.dimshuffle((0,'x'))
            
            dEdFws = finalStructuredLayerOutShuffled * initdEdInshuffled
            
            
            finalStructuredLayerdOutdIn = T.gt(finalStructuredLayerOut, T.zeros_like(finalStructuredLayerOut))
            finalStructuredLayerdOutdIn = T.cast(finalStructuredLayerdOutdIn, 'float32')
            
            finalStructuredLayerdEdOut = T.dot(initdEdIn, self.fws.T)
            
            finalStructuredLayerdEdIn = finalStructuredLayerdOutdIn * finalStructuredLayerdEdOut
            #up to here is correct

            newdEdIn = T.fvector('newdEdIn')
            dEdIns = [finalStructuredLayerdEdIn]
            newdEdIn = finalStructuredLayerdEdIn
            for i in reversed(range(self.hiddenlayers)):
                newdEdWs, newdEdBs, dEdIn = gradLayer(i, newdEdIn, outputs[i])
                newdEdIn = dEdIn
                dEdWs.append(newdEdWs)
                dEdBs.append(newdEdBs)
                dEdIns.append(newdEdIn)
            
            return predfinal, list(reversed(dEdWs)), list(reversed(dEdBs)), dEdFws, dEdFbs
#             return dEdIns
            
            
            
            
            
            
            

        def layer(ind, x):
            x = T.repeat(x, self.bMult)
#             print "in layer"
#             consolinds = self.consolidations[ind].inds
#             consols = self.consolidations[ind].complete
            if self.weightSharing[ind]:
              x *= T.tile(self.weights[ind], x.size//self.bMult)
            else:
              x*= self.weights[ind]
            
            x = T.concatenate([x, T.zeros((1), 'float32')])
            x = (x.take(self.consolMasks[ind]).astype(theano.config.floatX))
            x = T.dot(x, self.oneMultipliers[ind])
            
# #             inp, _ = theano.map(fn = lambda i: inp[i],
# #                                sequences = consols)
# #             #consolidation code
#             r = T.arange(consolinds.size - 1).astype(theano.config.floatX)
#             for i in xrange(len(self.nnet.consolidations[ind])):
#                 r = T.set_subtensor(r[i], x[consolinds[i]:consolinds[i + 1]].sum())
#             numConsolidatedRange = T.arange(consolinds.size - 1).astype('int32')
# #             inp2, _ = theano.map(fn = lambda i: inp1[consolinds[i]: consolinds[i + 1]].sum(),
# #                                                 sequences = numConsolidatedRange)
#             r, _ = theano.map(fn = lambda i, x: x[consolinds[i]: consolinds[i+1]].sum(),
#                                                 sequences = numConsolidatedRange,
#                                                 non_sequences=[x])
            if self.regression:
                return theano.tensor.nnet.relu(x + self.biases[ind])
            else:
                return T.tanh(x + self.biases[ind])
#             return r
            
#        = layer(0, inp)
        final = T.fvector()
        def loop(z):
            ret = T.fvector()
            for i in range(self.hiddenlayers):
                z = layer(i, z)
                ret = z
            return ret
        finals = loop(inp)
        final = T.dot(finals, self.fws)
        final = final + self.fbs
        if regression:
            final = theano.tensor.nnet.relu(final)
        else:
            final = T.nnet.softmax(final)
        
        self.classify = theano.function([inp], finals)
#         print self.classify(np.array([0,1,2,3]).astype(theano.config.floatX))

        index = T.iscalar()

        prediction, wgrads, bgrads, fwsgrads, fbsgrads = myGrad(inp, actualOutp)
#         mygradients = myGrad(inp, actualOutp)
#         self.realGrad = theano.function(inputs = [inp, actualOutp], outputs = list(reversed(gradients)))
    
        gradients = wgrads + bgrads + [fwsgrads, fbsgrads]
        param_Updates = OrderedDict((p, p - alpha * g) for p, g in zip(self.params, gradients))
        self.train = theano.function(inputs = [index, alpha],
                                                                    outputs = prediction,
                                                                    updates = param_Updates,
                                                                    givens = {
                                                                        inp : self.sharedTrainingInps[index],
                                                                        actualOutp: self.sharedTrainingOutps[index]
                                                                    })
        self.test = theano.function(inputs = [index],
            outputs = final,
            givens = {
                inp : self.sharedTestingInps[index]
            })
    
    def descend(self, alpha, epochs, trainingInps, trainingOutps, testingInps, testingOutps, verbose, stopNoImprovement = False):
        testLen = len(testingOutps)
        print "testLen is " + str(testingOutps[0])
        trainLen = len(trainingInps)
        indorder = range(trainLen)
        testingAccuracies = []



        start = time.time()
        testingAccuracy = 0.0
        if not self.regression:
            for j in range(len(testingOutps)):
                logloss = -np.dot(testingOutps[j], np.log(self.test(np.asscalar(np.array([j]).astype('int32'))))[0])
                testingAccuracy += logloss
            
                
        else:
            for j in range(testLen):
                testingAccuracy += LA.norm(self.test(np.asscalar(np.array([j]).astype('int32'))) - testingOutps[j])
        
        percentTestingAccuracy = None
        if self.regression:
            percentTestingAccuracy = testingAccuracy / (testLen * 1.0) * 100.0
        else:
            percentTestingAccuracy = testingAccuracy / (testLen * 1.0)
        testingAccuracies.append(percentTestingAccuracy)
        
        end = time.time()
        if verbose:
            print "epoch  0 -- testing accuracy : " + str(percentTestingAccuracy) + " duration: " + str(end - start) + "s"


        
        for i in range(epochs):
            print "starting epoch... testLen: " + str(testLen)
            print "test prediction for first test input: " + str(self.test(np.asscalar(np.array([0]).astype('int32'))))
            start = time.time()
            shuffle(indorder)
            for j in indorder:
                self.train(np.asscalar(np.array([j]).astype('int32')), np.asscalar(np.array([alpha]).astype('float32')))


            testingAccuracy = 0.0
            if not self.regression:
                for j in range(len(testingOutps)):
                    logloss = -np.dot(testingOutps[j], np.log(self.test(np.asscalar(np.array([j]).astype('int32'))))[0])
                    testingAccuracy += logloss
                
                    
            else:
                for j in range(testLen):
                    testingAccuracy += LA.norm(self.test(np.asscalar(np.array([j]).astype('int32'))) - testingOutps[j])
            
            percentTestingAccuracy = None
            if self.regression:
                percentTestingAccuracy = testingAccuracy / (testLen * 1.0) * 100.0
            else:
                percentTestingAccuracy = testingAccuracy / (testLen * 1.0)
            testingAccuracies.append(percentTestingAccuracy)
            end = time.time()
            if verbose:
                print "epoch " + str(i + 1) + " -- testing accuracy : " + str(percentTestingAccuracy) + " duration: " + str(end - start) + "s"

            if stopNoImprovement:
                if self.regression and max(testingAccuracies) == testingAccuracies[0]:
                    print "stopping from lack of improvement"
                    return max(testingAccuracies), testingAccuracies
                elif min(testingAccuracies) == testingAccuracies[0]:
                    print "stopping from lack of improvement"
                    return min(testingAccuracies), testingAccuracies


        if self.regression:
            return max(testingAccuracies), testingAccuracies
        else:
            return min(testingAccuracies), testingAccuracies