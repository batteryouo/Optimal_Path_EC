import math

import numpy as np

import objective

class Individual():

    minMutRate = 1e-100
    maxMutRate = 1
    learningRate = None
    uniprng = None
    normprng = None
    Obj = None

    def __init__(self):
        self.objectives = self.__class__.Obj(self.state)
        self.mutRate = self.uniprng.uniform(0.9,0.1) #Use "normalized" sigma
        self.numObj = len(self.objectives)
        self.frontRank = None
        self.crowdDist = None
        
    def dominates(self, other):
        '''
        It's your job to implement this function.
        Refers to HW7 but be careful that we're going to minimize the ManaCost while maximize the damage.
        (In HW7 we have already learned how to do the min-min problem, and now we are addressing the min-max problem.)
        '''

        if self is other:
            return 0

        selfBetter = 0
        otherBetter = 0

        if self.objectives[0] < other.objectives[0]:
            selfBetter += 1
        elif other.objectives[0] < self.objectives[0]:
            otherBetter += 1
        
        if self.objectives[1] > other.objectives[1]:
            selfBetter += 1
        elif other.objectives[1] > self.objectives[1]:
            otherBetter += 1

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
        """
        Compute distance between self & other in objective space
        """
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