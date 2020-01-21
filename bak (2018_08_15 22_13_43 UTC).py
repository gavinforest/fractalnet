from theano import shared,config,function
import itertools
import theano.tensor as T
import numpy as np
from collections import OrderedDict

def flatten(mylist):
    return list(itertools.chain.from_iterable(mylist))

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
    

def relu(x):
    return T.maximum(x, 0)


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
getSubV2Check()
def getV2Length(inds):
    return inds.size -1

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
    irange = T.arange(getV2Length(consolinds))
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


    
from random import shuffle

class tnnet:
    def __init__(self, resolution, functions, inputdimension, datainps, dataoutps, synapseThreshold, net = None):
        
        mynet = None
        if net == None:
            mynet = nnet(resolution, functions, inputdimension, datainps, dataoutps, synapseThreshold)
            mynet.genConsolidate()
            mynet.genWeights()
        else:
            mynet = net
        self.nnet = mynet
        
#         print "weights: " + str(mynet.weights)
#         print "biases: " + str(mynet.biases)
#         print mynet.feedforward(np.array([0,1,2,3]).astype(theano.config.floatX))
        inp_lengths = [len(consol) for consol in mynet.consolidations]
        inp_lengths = [datainps[0].size] + inp_lengths
        self.inp_lengths = np.array(inp_lengths).astype('int32')
#         print "inp_lengths: " + str(inp_lengths)
#         print "dinp length: " + str(datainps[0].size)
#         print "consolidations: " + str(mynet.consolidations)
        
        self.maxInp = np.amax(inp_lengths)
        self.maxPad = shared(np.array([0] * (np.amax(inp_lengths) - min(inp_lengths))).astype(theano.config.floatX))
        self.maxZeroes = shared(np.array([0.0] * self.maxInp).astype(theano.config.floatX))
        
        initialPaddingLength = np.amax(inp_lengths) - datainps[0].size
        self.initialPadding = shared(np.array([0.0] * initialPaddingLength).astype(theano.config.floatX))
        self.inp_lengths = shared(self.inp_lengths)
        
        tweights = V2IndexedShared(theano.config.floatX)
        tweights.fromList(mynet.weights)
        
        tbiases = V2IndexedShared(theano.config.floatX)
        tbiases.fromList(mynet.biases)
#         print tbiases.complete.get_value()
        
        tcons = V3IndexedShared(mynet.consolidations, 'int32')
        tfws = shared(mynet.finalweights)
        tfbs = shared(mynet.finalbiases)
        self.weightnum = mynet.weightnum
        self.hiddenlayers = mynet.hiddenlayers
        
        
        
        inp = T.fvector('inp')
        paddedInp = T.concatenate([inp, self.initialPadding])
        actualOutp = T.fvector('actualOutp')
        alpha = T.fscalar('alpha')
        
        self.winds = tweights.inds
        self.ws = tweights.complete
        self.binds = tbiases.inds
        self.bs = tbiases.complete
        self.consindinds = tcons.D3Inds
        self.consinds = tcons.D2Inds
        self.cons = tcons.complete
        self.bMult = shared(np.asscalar(np.array([mynet.branchMultiplier]).astype('int32')))
        self.fws = tfws
        self.fbs = tfbs

        self.params = [self.ws, self.bs, self.fws, self.fbs]
        self.constparams = [self.consindinds, self.consinds, self.cons, self.winds, self.binds, self.bMult]
        
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
        def layer(ind, inpLength, inp):
            consolinds = self.consinds[self.consindinds[ind] : self.consindinds[ind + 1] + 1]
            consols = self.cons[consolinds[0]: consolinds[consolinds.size - 1]]
            
            inp = T.repeat(inp[0:inpLength], self.bMult) * self.ws[self.winds[ind]:self.winds[ind + 1]]
            
            inp1 = T.choose(consols, inp)
#             inp1, _ = theano.map(fn = lambda i: inp[i],
#                                sequences = self.cons[consolinds[0]: consolinds[-1]])
#             #consolidation code
            numConsolidatedRange = T.arange(consolinds.size - 1).astype('int32')
            inp2, _ = theano.map(fn = lambda i: inp1[consolinds[i]: consolinds[i + 1]].sum(),
                                                sequences = numConsolidatedRange)
            inp3 = relu(inp2 + self.bs[self.binds[ind]:self.binds[ind+1]])
            padding = self.maxPad[0 : self.maxInp - inp3.size]
            return T.concatenate([inp3, padding])
           
        
        for i in range(self.hiddenlayers):
            inp = layer(i, self.inp_lengths[i], inp)
        
        
        final = T.dot(inp, self.fws)
        final = final + self.fbs
        final = relu(final)
        
        self.classify = theano.function([inp], final)
#         print self.classify(np.array([0,1,2,3]).astype(theano.config.floatX))
        
#         diff = final - actualOutp
#         squaredError = T.dot(diff, diff.T)
        
#         gradients = T.grad(squaredError, self.params, consider_constant = self.constparams)
#         param_Updates = OrderedDict((p, p - alpha * g) for p, g in zip(self.params, gradients))
#         self.train = theano.function(inputs = [inp, actualOutp, alpha],
#                                                                     outputs = squaredError,
#                                                                     updates = param_Updates)
                                                                    
    
    def descend(self, alpha, epochs, trainingInps, trainingOutps, testingInps, testingOutps, verbose):
        training = zip(trainingInps, trainingOutps)
        testing = zip(testingInps, testingOutps)
        testLen = len(testing)
        
        testingAccuracies = []
        
        for i in range(epochs):
            print "starting epoch..."
            start = time.time()
            shuffle(training)
            for (trainInp, trainOutp) in training:
                self.train(trainInp, trainOutp, alpha)
            
            testingAccuracy = 0.0
            for (testInp, testOutp) in testing:
                if np.argmax(self.classify(testInp)) == np.argmax(testOutp):
                    testingAccuracy += 1.0
                    
            percentTestingAccuracy = testingAccuracy / testLen * 100.0
            testingAccuracies.append(percentTestingAccuracy)
            end = time.time()
            if verbose:
                print "epoch " + str(i) + " -- testing accuracy : " + str(percentTestingAccuracy) + " duration: " + str(start - end)
        
        return min(testinAccuracies), testingAccuracies
            
        
tn = tnnet(10, [[lambda x: x + 1], [lambda x: x * 2]], 1, [np.array([0,1,2,3])], [np.array([1])], 100)
# tn.nnet.feedforward(np.array([0,1,2,3]))
tn.classify(np.array([0,1,2,3]).astype(theano.config.floatX))