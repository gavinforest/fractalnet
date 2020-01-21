import operator
import math
from deap import gp, creator, base, tools, algorithms
from random import randint, uniform
import pickle
import itertools
import time
import math
from itertools import izip
import numpy as np

DataDims = 1
MINDIMFUNCS = 8
MAXDIMFUNCS = 40
MINDIMS = DataDims
MAXDIMS = 2
MU = 40
LAMBDA = 30
CXPB = 0.5
MUTPB = 0.3

unif = lambda: uniform(-1.0,1.0)

def square(x):
    return x*x
def absSqrt(x):
    return math.sqrt(abs(x))
pset = gp.PrimitiveSetTyped("nodePrims", [float], float)
pset.addEphemeralConstant("nprandEph", unif, float)
pset.addPrimitive(max, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(absSqrt, [float], float)
pset.addPrimitive(square, [float], float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,) )
creator.create("Individual", list, fitness=creator.FitnessMin)
# def myMutate(individual):
#     new = [[gp.mutNodeReplacement(x,pset=pset) for x in l] for l in toolbox.clone(individual)]
#     return (tools.initIterate(creator.Individual, lambda: new),)


def getFitMap(ls):
    fitnesses = []
    for l in ls:
        fitnesses.append(l.fitness.values[0])
    return fitnesses


class evolution():

    def __init__(self, nnetEvaluator, generations, popsize, pool = None):

        self.pool = pool

        self.pset = pset

        def genTreeList():
            return [gp.PrimitiveTree(gp.genHalfAndHalf(pset=self.pset, min_=1,max_=6)) for i in range(randint(MINDIMFUNCS,MAXDIMFUNCS))]

        self.genTreeList = genTreeList

        def genFuncList():
            return [genTreeList() for j in range(randint(MINDIMS,MAXDIMS))]
        self.genFuncList = genFuncList

        self.evaluator = nnetEvaluator
        self.generations = generations
        self.popsize = popsize


    def evolve(self):
        
        
        
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,) )
        # creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.genFuncList)
        
        def myMutate(individual):
            new = [[gp.mutNodeReplacement(x,pset=self.pset) for x in l] for l in toolbox.clone(individual)]
            new = [[x[0] for x in l] for l in new]
            return (tools.initIterate(creator.Individual, lambda: new),)

        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", myMutate)
        toolbox.register("select",tools.selTournament, tournsize = 4)
        toolbox.register("evaluate", self.evaluator)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(self.popsize)

        if self.pool is not None:
            toolbox.register("map", self.pool.map)
        

        myStats = tools.Statistics()
        myStats.register("mean", lambda ls: np.mean(getFitMap(ls)))
        myStats.register("min", lambda ls: min(getFitMap(ls)))
        myStats.register("max", lambda ls: max(getFitMap(ls)))
        myStats.register("stdDev", lambda ls: np.std(getFitMap(ls)))

        hallOFame = tools.HallOfFame(5) # hall of fame of size 5


        (finalPop, logbook) = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, self.generations, myStats, halloffame=hallOFame, verbose=True)

        gen = logbook.select("gen")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("mean")

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_maxs, "b-", label="Maximum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, fit_avgs, "r-", label="Average Fitness")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.show()
        return hallOFame

