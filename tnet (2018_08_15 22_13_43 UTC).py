from random import shuffle

class tnnet:
    def __init__(self, nwinds, nws, nbinds, nbs, nconsindinds, nconsinds, ncons, nfws, nfbs, nbMult):
        
        inp = T.fvector('inp')
        actualOutp = T.fvector('actualOutp')
        alpha = T.fscalar('alpha')
        
        self.winds = shared(nwinds)
        self.ws = shared(nws)
        self.binds = shared(nbinds)
        self.bs = shared(nbs)
        self.consindinds = shared(nconsindinds)
        self.consinds = shared(nconsinds)
        self.cons = shared(ncons)
        self.bMult = shared(nbMult)
        self.fws = shared(nfws)
        self.fbs = shared(nfbs)

        self.params = [self.ws, self.bs, self.fws, self.fbs]
        
        
        def layer(ind, inp):
            weights = getSubV2(ind, winds, ws)
            biases = getSubV2(ind, binds, bs)
            consolinds, consols = getSubV3(ind, consindinds, consinds, cons)
            inp = T.repeat(inp, bMult)
            inp = weights * inp
            inp = consolidateTensor(inp, consolinds, consols)
            inp = relu(inp + biases)
            
        irange = T.arange(consindinds.size - 1)
        final, _ = theano.scan(fn = layer,
                                                        sequences = irange,
                                                        outputs_info = inp,
                                                        n_steps = irange.shape[0]
                                                        )
        
        final = final * fws
        final = final + fbs
        final = relu(final)
        
        self.classify = theano.function([inp], final)
        
        diff = outp - actualOutp
        squaredError = (diff * diff).sum()
        
        gradients = T.grad(squaredError, self.params)
        param_Updates = OrderedDict((p, p - alpha * g) for p, g in zip(self.params, self.gradients))
        self.train = theano.function(inputs = [inp, actualOutp, alpha],
                                                                    outputs = squaredError,
                                                                    updates = param_Updates)
                                                                    
    
    def descend(alpha, epochs, trainingInps, trainingOutps, testingInps, testingOutps, verbose):
        training = zip(trainingInps, trainingOutps)
        testing = zip(testingInps, testingOutps)
        testLen = len(testing)
        for i in range(epochs):
            training = shuffle(training)
            for (trainInp, trainOutp) in training:
                self.train(inp, outp, alpha)
            
            testingAccuracy = 0.0
            for (testInp, testOutp) in testing:
                if np.argmax(self.classify(testInp)) == np.argmax(testOutp):
                    testingAccuracy += 1.0
                    
            percentTestingAccuracy = testingAccuracy / testLen * 100.0
            
            if verbose:
                print "epoch " + str(i) + " -- testing accuracy : " + str(percentTestingAccuracy)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# class tNet:
#     def __init__(nwinds, nws, nbinds, nbs, nconsindinds, nconsinds, ncons, nfws, nfbs, nbMult)
        
#         winds = shared(nwinds)
#         ws = shared(nws)
#         binds = shared(nbinds)
#         bs = shared(nbs)
#         consindinds = shared(nconsindinds)
#         consinds = shared(nconsinds)
#         cons = shared(ncons)
#         fws = shared(fws)
#         fbs = shared(nfbs)
#         bMult = shared(nbMult)
        
#         inp = T.fvector('inp')
        
        
        
#         def layer(ind, inp):
#             weight = getSubV2(ind, winds, ws)
#             bias = getSubV2(ind, binds, bs)
#             tconsolinds, tconsols = getSubV3(ind, consindinds, consinds, cons)
            
#             inp = T.repeat(inp, branchMultiplier)
#             inp = inp * weights
#             inp = consolidateTensor(inp, tconsolinds, tconsols)
#             inp = relu(inp + biases)
            
#         irange = T.arange(consindinds.size - 1)
#         final, updates = theano.scan(fn = theanoScanFunc,
#                                      sequences = [irange],
#                                      outputs_info = binp,
#                                      non_sequences = [winds,ws,binds,bs,consindinds,consinds,cons,bMult])
#         return relu(T.dot(fws, final) + fbs)
            
        

# def layer(inp, weights, biases, consolinds, consolcomplete, branchMultiplier):
#     inp = T.repeat(inp, branchMultiplier)
#     inp = inp * weights
#     inp = consolidateTensor(inp, consolinds, consolcomplete)
#     inp = relu(inp + biases)

# def layerCheck():
#     x, w, b = T.vectors('x', 'w', 'b')
#     consinds, conscomp = T.ivectors('inds', 'complete')
#     branchMultiplier = T.iscalar('multiplier')
#     f = theano.function([x,w,b,consinds,conscomp, branchMultiplier], layer(x, w, b, consinds, conscomp, branchMultiplier), allow_input_downcast=True)
#     print f(np.array([1]), np.array([2,2,2]), np.array([5]), np.array([0,3]), np.array([0,1,2]), 3)
    
# def theanoScanFunc(ind, inp, weightinds, weights, biasinds, biases, consolindinds, consolinds, consols, branchMultiplier):
#     weight = getSubV2(ind, weightinds, weights)
#     bias = getSubV2(ind, biasinds, biases)
#     tconsolinds, tconsols = getSubV3(ind, consolindinds, consolinds, consols)
#     return layer(inp, weight, bias, tconsolinds, tconsols, branchMultiplier)
    
# def feed(winds, ws, binds, bs, consindinds, consinds, cons, fws, fbs, bMult):
    
#     irange = T.arange(consindinds.size - 1)
#     final, updates = theano.scan(fn = theanoScanFunc,
#                                  sequences = [irange],
#                                  outputs_info = binp,
#                                  non_sequences = [winds,ws,binds,bs,consindinds,consinds,cons,bMult])
#     return relu(T.dot(fws, final) + fbs)

# def theanoScanFuncCheck():
#     winds, binds = T.ivectors('winds', 'binds')
#     inp, w, b = T.vectors('inp', 'w', 'b')
#     consindinds, consinds, conscomp = T.ivectors('indinds', 'inds', 'complete')
#     mult = T.iscalar('multiplier')
#     ind = T.iscalar('ind')
    
#     ninp = np.array([1]).astype(theano.config.floatX)
#     nw = np.array([1,1,1,2,2,2]).astype(theano.config.floatX)
#     nwinds = np.array([0,3,6]).astype('int32')
#     nb = np.array([1,2,2]).astype(theano.config.floatX)
#     nbinds = np.array([0,1,3]).astype('int32')
#     ncons = np.array([0,1,2,0,1,2]).astype('int32')
#     nconsinds = np.array([0,3,4,6]).astype('int32')
#     nconsindinds = np.array([0,2,3]).astype('int32')
#     nmult = np.asscalar(np.array([3]).astype('int32'))
#     nind = np.asscalar(np.array([1]).astype('int32'))
    
#     f = theano.function([ind, inp, winds, w, binds, b, consindinds, consinds, conscomp, mult],
#                         theanoScanFunc(ind, inp, winds, w, binds, b, consindinds, consinds, conscomp, mult))
#     print f(nind, ninp, nwinds, nw, nbinds, nb, nconsindinds, nconsinds, ncons, nmult)

# def feedCheck():
#     twinds, tbinds = T.ivectors('winds', 'binds')
#     tinp, tw, tb = T.vectors('inp', 'w', 'b')
#     tconsindinds, tconsinds, tconscomp = T.ivectors('indinds', 'inds', 'complete')
#     tmult = T.iscalar('multiplier')
#     tfbs = T.vector('fbs')
#     tfws = T.matrix()
    
#     ninp = np.array([1]).astype(theano.config.floatX)
#     nw = np.array([1,1,1,2,2,2]).astype(theano.config.floatX)
#     nwinds = np.array([0,3,6]).astype('int32')
#     nb = np.array([1,2,2]).astype(theano.config.floatX)
#     nbinds = np.array([0,1,3]).astype('int32')
#     ncons = np.array([0,1,2,0,1,2]).astype('int32')
#     nconsinds = np.array([0,3,4,6]).astype('int32')
#     nconsindinds = np.array([0,2,3]).astype('int32')
#     nmult = np.asscalar(np.array([3]).astype('int32'))
#     nfws = np.array([[1,1],[1,1]]).astype(theano.config.floatX)
#     nfbs = np.array([1,1]).astype(theano.config.floatX)
    
#     f = theano.function([tinp, twinds, tw, tbinds, tb, tconsindinds, tconsinds, tconscomp, tfws, tfbs, tmult],
#                         feed(tinp, twinds, tw, tbinds, tb, tconsindinds, tconsinds, tconscomp, tfws, tfbs, tmult))
#     print f(ninp, nwinds, nw, nbinds, nb, nconsindinds, nconsinds, ncons, nfws, nfbs, nmult)
    
#     def consolidate(tensors, consolidationList): #for theano
#         for sublist in consolidationList:
#             for i in sublist[1:]:  #iterate from second to last element backwards
#                 #increment tensor specified by sublist[0] at position [-1] by the [-1] element of the other tensors in sublist
#                 tensors[sublist[0]].inc_subtensor(tensors[sublist[0]][-1], tensors[sublist[i]][-1]) #use last element = activation
#                 del tensors[sublist[i]] #delete tensor that was added to sublist[0] tensor

#         return tensors

#     def positionConsolidate(narrays, consolidationList): #for numpy
#         ret = []
#         for sublist in consolidationList:
#             ret.append(narrays[sublist[0]])

#         return ret

#         #this is only needed when generating consolidation arrays. Otherwise, can just use elements without spacial coordinates



#         elif self.inputdimension == 2:
#             if not self.inputdims:
#                 factors = ((i, insize/i) for i in range(int(insize**0.5),0,-1)
#                             if insize % i == 0)
#                 self.inputdims = factors.next()

#             x,y = self.inputdims
#             for i in range(x):
#                 for j in range(y):
#                     myTens = tensorframe
#                     myTens[0] = (i * 1.0) / x
#                     myTens[1] = (j * 1.0) / y
#                     located.append(myTens)

#         return located #returns list of numpy arrays in same order as sample

#     def genweights(self, indimension, sampleinput, outsize, threshold):
#         myArrays = self.locate(sampleinput)
#         myWeightNum = 0
#         self.hiddenlayers = 0
#         while myWeightNum < threshold:
#             new = self.applyFuncsMult(myArrays)

#             weightVec = shared(np.random.rand(len(new)))
#             biasVec   = shared(np.random.rand(len(new)))
#             self.weights.append(weightVec)
#             self.biases.append(biasVec)
#             myWeightNum += len(new)

#             self.consolidations.append(genConsolidate(new))
#             myArrays = numpyConsolidate(new, self.consolidations[-1])

#             #convert to shared data type for later speed
#             self.consolidations[-1] = [shared(subArray) for subArray in self.consolidations[-1]] 
#             self.hiddenlayers += 1

#         self.weights.append(shared(
#             np.random.rand(len(mytensors),outsize)))        #final interconnected layer for output
#         self.biases.append(shared(
#             np.random.rand(len(mytensors))))

#         myWeightNum += len(mytensors) * outsize
#         self.weightNum = myWeightNum

#     def feedForward(self, inp, weights, biases, consolidations, hiddenlayers):
#         inputs = T.ravel(inp)

#         for i in range(hiddenlayers):
#             new = 
#             new = new * weights[i] + biases[i]
#             new = self.consolidate(new, consolidations[i])
#             new = [T.tanh(tensor) for tensor in new]
#             inputs = new

#         inputs = theano.dot(inputs, self.weights[-1])
#         inputs += self.biases[-1]
#         return T.tanh(inputs)

#     def error(selif,inp, outp):
#         diff = outp - self.feedForward(inp)
#         return T.sum(T.dot(diff, diff))

#     def train(self, num_epochs):

#         ind = T.iscalar('index')
#         x = T.tensor('input')
#         y = T.tensor('output')
#         err = T.scalar(self.error(x,y))
#         error = theano.function([x,y], err)

#         gw = [theano.grad(error,weight) for weight in self.weights]
#         gb = [theano.grad(error,bias)   for bias in self.biases]
#         updates = [(weight, weight - wgrad * alpha) for weight, wgrad in zip(self.weights, gw)] +\
#                   [(bias  , bias  -  bgrad * alpha) for bias ,  bgrad in zip(self.biases , gb)]
#         train = theano.function(
#                                 inputs = [ind],
#                                 outputs = err,
#                                 updates = updates,
#                                 given = {
#                                     x : self.dataset[ind],
#                                     y : self.datalabels[ind] }
#                                 )
#         print '... training'
#         for i in range(num_epochs):
#             errors = []
#             for ind in range(len(self.datalabels)):
#                 errors.append(train(ind))
#             print "epoch " + str(i) + " complete, error is " + sum(errors)/len(self.datalabels)

#         len(logbook.select('gen'))
            
        