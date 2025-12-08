import math

import numpy as np

from . import objective
from . import func
from . import shape
from . func import motion

class Individual():

    uniprng = None
    normprng = None
    objective_func = None
    constrain_func = None

    def __init__(self, minMutRate = 1e-100, maxMutRate = 1, learningRate = 1e-1):

        self.objectives = objective.MultiObjective(objectives_func_list=self.__class__.objective_func, **self.states)
        if self.constrain_func is not None:
            self.constrains = func.MultiConstrain(self.constrain_func, self.states)
        else:
            self.constrains = None
        self.mutRate = self.uniprng.uniform(0.1,0.9) #Use "normalized" sigma
        self.numObj = len(self.objectives)
        self.frontRank = None
        self.crowdDist = None
        
    def dominates(self, other, compare_list = ["max", "max"]):

        if self is other:
            return 0
        
        if self.constrains is not None and other.constrains is not None:
            selfConstrain = np.sum(self.constrains.results)
            otherConstrain = np.sum(other.constrains.results)
            
            if selfConstrain > otherConstrain:
                return 1
            if selfConstrain < otherConstrain:
                return -1
            if selfConstrain != 0:
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
    def __init__(self, img, pts):
        states = []
        self.img = img
        self.pts = pts
        self.pathLine = []

        model = motion.ConstMotion(5)      

        for i in range(len(pts) - 1):
            line = shape.Line(pts[i], pts[i + 1])
            self.pathLine.append(line)
        self.pathLine.append(shape.Line(pts[-1], pts[0]))

        firstPointIn = self.uniprng.uniform(0, 1)
        theta = model.findTheta(self.pathLine[0].percentage2point(firstPointIn), self.pathLine[1].a, self.pathLine[1].b, self.pathLine[1].c,
                                self.pathLine[0].theta, self.pathLine[1].theta)
        
        if theta is not None:
            lastPointOut = model.calEndXY(self.pathLine[0].percentage2point(firstPointIn), theta, self.pathLine[0].theta, self.pathLine[1].theta)
            lastPointOut = np.linalg.norm(lastPointOut - pts[1]) / self.pathLine[1].length
        else:
            lastPointOut = 0
        states.append([firstPointIn, lastPointOut])

        for _ in range(len(self.pts) - 2):
            states.append([self.uniprng.uniform(lastPointOut, 1), self.uniprng.uniform(0, 1)])
            lastPointOut = states[-1][1]
        
        states.append([self.uniprng.uniform(lastPointOut, 1), self.uniprng.uniform(0, firstPointIn)])


        for i in range(len(pts) - 1):
            theta = model.findTheta(self.pathLine[i].percentage2point(states[i][0]), self.pathLine[i+1].a, self.pathLine[i+1].b, self.pathLine[i+1].c,
                            self.pathLine[i].theta, self.pathLine[i + 1].theta)
            print(theta)
                                
        
        self.states = {"states": states}
        super().__init__()
