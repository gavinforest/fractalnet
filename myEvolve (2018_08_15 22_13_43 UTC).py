import operator
import math
from deap import gp, creator, base, tools, algorithms
from random import randint, uniform
import random
import pickle
import itertools
import time
import math
from itertools import izip
import numpy as np
import copy

constantLims = (-5.0, 5.0)
resolutionLimits = (25,100)
resModifyLimits = (-5,5)
resModifyRange = (0,5)
dimAddingRange = (0,5)
alphaModifyRange = (0,5)
alphaModifyMult = (0.5,1.5)
weightSharingPb = 0.3
weightSharingModPb = 0.2

DataDims = 1
MINDIMFUNCS = 8
MAXDIMFUNCS = 20
MINDIMS = DataDims
MAXDIMS = 5
MINFUNCDEPTH = 2
MAXFUNCDEPTH = 5
CXPB = 0.7
MUTPB = 0.3
MUTSTRENGTH = 0.3
NAZIFACTOR = 0.5
SUDDENDEATHTOURNEYSIZE = 5
FitnessMaximize = -1.0
defaultAlphaRange = (0.001, 0.01)

unif = lambda: uniform(*constantLims)

def square(x):
    return x*x
def absSqrt(x):
    return math.sqrt(abs(x))
    
singletons = [math.sin, math.cos, square, absSqrt]
doubles = [operator.mul, operator.add, operator.sub]

class Tree:
    def __init__(self, depth):
        self.depth = depth
        
        if depth > 0: #going to branch
            if random.choice([True, False]):
                self.single = True
                self.op = random.choice(singletons)
                self.below = Tree(self.depth - 1)
            else:
                self.single = False
                self.op = random.choice(doubles)
                self.left = Tree(self.depth -1)
                self.right = Tree(self.depth - 1)

        else: #going to be a constant or a variable
            if random.choice([True, False]):
                self.isConstant= True
                self.constant = unif()
            else:
                self.isConstant = False
            
    
    def __call__(self, x):
        if self.depth == 0:
            if self.isConstant:
                return self.constant
            else:
                return x
        elif self.single:
            return self.op(self.below(x))
        else:
            return self.op(self.left(x), self.right(x))
            



class Individual:
    def __init__(self):
        frameLength = [None] * random.randint(MINDIMS, MAXDIMS)
        lengthDimFuncs = [random.randint(MINDIMFUNCS, MAXDIMFUNCS) for i in frameLength]
        funcDepths = [[random.randint(MINFUNCDEPTH, MAXFUNCDEPTH) for i in range(j)] for j in lengthDimFuncs]
        self.funcs = [map(Tree, l) for l in funcDepths]
        self.resolution = random.randint(*resolutionLimits)
        self.alpha = random.uniform(*defaultAlphaRange)
        
        self.fitness = None
        self.fitnessValid = False
        self.weightSharing = []
        while random.random() < weightSharingPb:
          self.weightSharing.append(True)
        
    def Mutate(self):
        willAdd = random.choice([1,0,-1])
        willModDim = random.choice([True, False])
        mutatedFuncs  = []
        mutatedAlpha = self.alpha
        mutatedRes = self.resolution

        #mutating functions
        for funcList in self.funcs:
            mutatedFuncList = []
            for f in funcList:
                if random.random() < MUTSTRENGTH:
                    mutatedFuncList.append(Tree(random.randint(MINFUNCDEPTH, MAXFUNCDEPTH)))
                else:
                    mutatedFuncList.append(f)
            mutatedFuncs.append(mutatedFuncList)

        if willAdd == 1:
            if willModDim:
                mutatedFuncs = mutatedFuncs + [[Tree(random.randint(MINFUNCDEPTH, MAXFUNCDEPTH)) for i in range(random.randint(MINDIMFUNCS, MAXDIMFUNCS))]]
            else:
                ind = random.randint(0, len(mutatedFuncs)-1)
                mutatedFuncs[ind] = mutatedFuncs[ind] + [Tree(random.randint(MINFUNCDEPTH, MAXFUNCDEPTH))]
        elif willAdd == -1:
            if willModDim:
                mutatedFuncs.pop(random.randint(0, len(mutatedFuncs)-1))
            else:
                ind = random.randint(0, len(mutatedFuncs) - 1)
                mutatedFuncs[ind].pop(random.randint(0, len(mutatedFuncs[ind]) - 1))

        willMutateAlpha = random.uniform(*alphaModifyRange) < 1.0
        if willMutateAlpha:
            mutatedAlpha *= random.uniform(*alphaModifyMult)

        willMutateRes = random.uniform(*resModifyRange) < 1.0
        if willMutateRes:
            mutatedRes += random.randint(*resModifyLimits)
            if mutatedRes < resolutionLimits[0]:
                mutatedRes = resolutionLimits[0]
            elif mutatedRes > resolutionLimits[1]:
                mutatedRes = resolutionLimits[1]

        newWeightSharing = []
        for old in self.weightSharing:
          if random.random() > weightSharingModPb:
            newWeightSharing.append(old)
          else:
            newWeightSharing.append(not old)
        
        self.funcs = mutatedFuncs
        self.alpha = mutatedAlpha
        self.resolution = mutatedRes
        self.weightSharing = newWeightSharing
        self.fitnessValid = False
        
        return self

    def Mate(aMate):
        self.fitnessValid = False
        newFuncs = []
        cxlength = min(len(self.funcs), len(aMate.funcs))
        for i in range(cxlength):
            if random.choice([True, False]):
                newFuncs.append(self.funcs[i])
            else:
                newFuncs.append(aMate.funcs[i])

        if len(self.funcs) > len(aMate.funcs):
            newFuncs += self.funcs[cxlength:]
        elif len(self.funcs) < len(aMate.funcs):
            newFuncs += aMate.funcs[cxlength:]

        newRes = math.round((self.resolution + aMate.resolution) * 0.5)
        newAlpha = (self.alpha + aMate.alpha) * 0.5
        
        newWeightSharing = None
        if len(self.weightSharing) >= len(aMate.weightSharing):
          newWeightSharing = self.weighSharing
        else:
          newWeightSharing = aMate.weightSharing
        
        
        self.funcs = newFuncs
        self.resolution = newRes
        self.alpha = newAlpha
        self.weightSharing = newWeightSharing
        self.fitnessValid = False

        return self

def nazi(population):
    readyforGermany = len([x.fitnessValid for x in population if x.fitnessValid]) == len(population)
    europe = population
    aryans = []
    while len(aryans) < len(population) * (1.0 - NAZIFACTOR):
        random.shuffle(europe)
        tournamentCompetitors = europe[0:SUDDENDEATHTOURNEYSIZE]
        fitnesses = [x.fitness * FitnessMaximize for x in tournamentCompetitors]
        ind = fitnesses.index(max(fitnesses))
        ubermench = tournamentCompetitors[ind]
        aryans.append(ubermench)
        europe.pop(ind)

    return aryans





class evolution():

    def __init__(self, nnetEvaluator, generations, popsize, pool = None):

        self.pool = pool

        self.evaluator = nnetEvaluator
        self.generations = generations
        self.popsize = popsize


    def evolve(self):
        def printProgress(epoch, average, standardDeviation, minimum, maximum):
            print "epoch " + str(epoch) + " avg: " + str(average) + " std: " + str(standardDeviation) + " min,max -- " + str((minimum,maximum))
        
        self.population = [Individual() for i in range(self.popsize)]

        avgFitnesses = []
        fitnessStds = []
        minimums = []
        maximums = []
        
        for i in range(self.generations):
            if self.pool is None:
                self.population = map(self.evaluator, self.population)
            else:
                self.population = self.pool.map(self.evaluator, self.population)

            fitnesses = [x.fitness for x in self.population]


            avgFitness = np.mean(fitnesses)
            fitnessStd = np.std(fitnesses)
            amininum = min(fitnesses)
            amaximum = max(fitnesses)
            printProgress(i, avgFitness, fitnessStd, amininum, amaximum)
            avgFitnesses.append(avgFitness)
            fitnessStds.append(fitnessStd)
            minimums.append(amininum)
            maximums.append(amaximum)

            self.population = nazi(self.population)
            toGenerate = self.popsize - len(self.population)
            toGenMutate = int(round(toGenerate * MUTPB))
            toMate = int(round(toGenerate * CXPB))

            random.shuffle(self.population)
            self.population += [copy.deepcopy(x).Mate(y) for x,y in zip(self.population[:toMate], self.population[toMate: 2 * toMate])]

            random.shuffle(self.population)
            self.population += [copy.deepcopy(x).Mutate() for x in self.population[:toGenMutate]]

        print "evolution complete! Here's what happened"

        for i in range(len(avgFitnesses)):
            printProgress(i, avgFitnesses[i], fitnessStds[i], minimums[i], maximums[i])





        return self.population

