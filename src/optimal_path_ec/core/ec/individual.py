import math

import numpy as np

import objective

class Individual():

    minMutRate = 1e-100
    maxMutRate = 1
    learningRate = None
    uniprng = None
    normprng = None
    objective_func = None

    def __init__(self):
        self.objectives = objective.MultiObjective(self.__class__.objective_func, self.states)
        self.mutRate = self.uniprng.uniform(0.9,0.1) #Use "normalized" sigma
        self.numObj = len(self.objectives)
        self.frontRank = None
        self.crowdDist = None
        
    def dominates(self, other, compare_list = ["max", "max"]):

        if self is other:
            return 0
        
        compare_results = self.objectives.compareValue(other.objectives, compare_list)
        selfBetter = np.count_nonzero(compare_results)
        otherBetter = len(compare_results) - selfBetter
        
        if selfBetter > 0 and otherBetter == 0:
            return 1
        elif otherBetter > 0 and selfBetter == 0:
            return -1

        return 0
    
    def compareRankAndCrowding(self, other):
        '''
        It's your job to implement this function.
        You could utilize the function from your (or the answer's) HW7!
        '''
        if self.frontRank < other.frontRank:
            return 1
        elif other.frontRank < self.frontRank:
            return -1
        else:
            if self.crowdDist > other.crowdDist:
                return 1
            elif other.crowdDist > self.crowdDist:
                return -1
            else:
                return 0
                
    def distance(self, other, normalizationVec=[None]):

        # check if self vs self
        if self is other:
            return 0.0
        
        #set default normalization to 1.0, if not specified
        if normalizationVec[0] == None:
            normalizationVec=[1.0]*self.numObj
            
        # compute normalized Euclidian distance
        distance = 0
        i = 0
        while i < self.numObj:
            tmp=(self.objectives[i]-other.objectives[i])/normalizationVec[i]
            distance+=(tmp*tmp)
            i+=1
            
        distance=math.sqrt(distance)
        
        return distance

class PathIndividual(Individual):
    def __init__(self):
        super().__init__()
