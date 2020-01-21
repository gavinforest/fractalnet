import mnist
import numpy as np
import evolve
import fractalNetworkCPU as fractalNet
import deap.gp as gp
from functools import partial
import multiprocessing
import theano
import itertools
from theano import shared
import time
lock = multiprocessing.Lock()
from sklearn import preprocessing


#<numerai section>
import csv

trainingFilePath = "numerai_training_data.csv"
tournamentFilePath = "numerai_tournament_data.csv"

def readTraining(splitforTesting = True):
    trainingFile = open(trainingFilePath, 'rb')
    trainingFeatures = []
    trainingLabels = []

    rownum = 0
    reader = csv.reader(trainingFile)
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            features = row[:len(row) - 1 ]
            trainingFeatures.append(np.array(features).astype('float32'))
            trainingLabels.append(np.array([row[-1]]).astype('float32'))

                
        rownum += 1
    
    for i in range(len(trainingLabels)):
        if trainingLabels[i] == np.array([1.0]).astype('float32'):
            trainingLabels[i] = np.array([0.0,1.0]).astype('float32')
        else:
            trainingLabels[i] = np.array([1.0,0.0]).astype('float32')
    
    trainingFeatures = preprocessing.scale(np.array(trainingFeatures))
    trainingFeatures = [np.array(x).astype('float32') for x in trainingFeatures.tolist()]
    if splitforTesting:
        testingFeatures = trainingFeatures[len(trainingFeatures) - 10000 : len(trainingFeatures)]
        testingLabels = trainingLabels[len(trainingLabels) - 10000 : len(trainingLabels)]

        trainingFeatures = trainingFeatures[0:len(trainingFeatures) - 10000]
        trainingLabels = trainingLabels[0:len(trainingLabels) - 10000]
        return trainingFeatures, trainingLabels, testingFeatures, testingLabels

    else:
        return trainingFeatures, trainingLabels


def readTournament():
    tournamentFile = open(tournamentFilePath, 'w+')
    tournamentFeatures = []

    rownum = 0
    reader = csv.reader(tournamentFile)
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            features = row[:]
            tournamentFeatures.append(np.array(features).astype('float32'))

                
        rownum += 1

    return tournamentFeatures

#</numerai section>






NEURONTHRESHOLD = 130000
EPOCHS = 3
ALPHA = 0.003
RES = 12
REGRESSION = False

#must be customized to each new dataset
def labelToArray(x):
    blank = [0] * 10
    blank[x] = 1
    return np.array(blank)

def flatten(mylist):
    return list(itertools.chain.from_iterable(mylist))


def nnetEvaluator(individual, trainimgs, trainlabels, testimgs, testlabels):
    
    print "nnetEvaluator testingims length: " + str(len(testimgs))
    
    lock.acquire()
    print "evaluating"
    lock.release()
    funcs = []
    for dimlist in individual:
        newDimlist = []
        for tree in dimlist:
#             print gp.stringify(tree)
            f = gp.compile(tree, evolve.pset)
            newDimlist.append(f)
        funcs.append(newDimlist)
    
    tnet = fractalNet.tnnet(RES, funcs, 1, trainimgs, trainlabels, testimgs, testlabels, NEURONTHRESHOLD, REGRESSION)
    mynet = tnet.nnet
    lock.acquire()
    print "branchmultiplier: " + str(mynet.branchMultiplier)
    print "weights number: " + str(mynet.weightnum)
    print "consolidations lengths" + str(map(len, mynet.consolidations))
    layerDensities = []
    prevlayerLength = trainimgs[0].size
    for layer in mynet.weights:
        layerLength = len(layer)
        branched = prevlayerLength * mynet.branchMultiplier
        layerDensities.append((1.0 * branched) / layerLength)
        prevlayerLength = layerLength
    print "average number of connections per neuron by layer: " + str(layerDensities)
    print "spacial dimension: " + str(len(funcs))
    print "branching by layer: " + str(map(len, funcs))
    testStart = time.time()
    x = tnet.train(1, ALPHA)
    print "time to train once: " + str(time.time() - testStart)
    print "beginning training"
    print
    lock.release()
    # print "first test output: " + str(testlabels)
    maxAccuracy, avgpercenttestaccuracylist = tnet.descend(ALPHA, EPOCHS, trainimgs,  trainlabels, testimgs, testlabels, True)
    lock.acquire()
    print "net with branching of " + str(mynet.branchMultiplier) + " and " + \
        str(mynet.weightnum) + " had accuracies of: " + str(avgpercenttestaccuracylist)
    lock.release()

    return maxAccuracy,

def run():
    print "loading data into RAM..."
    # training = list(mnist.read(dataset = "training"))
    # testing = list(mnist.read(dataset = "testing"))
    print "reformatting data..."
    
    trainingfeats, traininglabels, testingfeats, testinglabels = readTraining(splitforTesting=True)
    # trainingimgs = [train[1].astype(theano.config.floatX).ravel() * (1.0/256) for train in training]
    # traininglabels = [labelToArray(train[0]).astype(theano.config.floatX) for train in training]
    # testingimgs = [test[1].astype(theano.config.floatX).ravel() * (1.0/256) for test in testing]
    # testinglabels = [labelToArray(test[0]).astype(theano.config.floatX) * (1.0 / 256) for test in testing]

    # trainingimgs = map(shared, trainingimgs)
    # traininglabels = map(shared, traininglabels)
    # testingimgs = map(shared, testingimgs)
    # testinglabels = map(shared, testinglabels)
    print len(testingfeats)
    print len(trainingfeats)
    print trainingfeats[0]
    print traininglabels[0]
    
    # p = multiprocessing.Pool(3)
    print "data reformatting complete. Beginning evolution. Training is " + str(len(trainingfeats)) + "examples, testing " + str(len(testingfeats))
    evaluator = partial(nnetEvaluator, trainimgs = trainingfeats[:], trainlabels = traininglabels[:],
                                        testimgs = testingfeats[:], testlabels = testinglabels[:])

    evolutionFramework = evolve.evolution(evaluator, 10, 10)
    evolutionFramework.evolve()

if __name__ == "__main__":
    run()
