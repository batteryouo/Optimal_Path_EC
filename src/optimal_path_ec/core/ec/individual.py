import math

import numpy as np

from . import objective
from . import func
from . import shape
from . func import motion

class Individual():

    uniprng = None
    normprng = None
    def __init__(self, objective_func = None, constrain_func = None, minMutRate = 1e-100, maxMutRate = 1, learningRate = 1e-1):
        self.objective_func = objective_func
        
        self.objective_func = objective_func
        self.constrain_func = constrain_func
        self.minMuteRate = minMutRate
        self.maxMuteRate = maxMutRate
        self.learningRate = learningRate
        
        self.objectives = objective.MultiObjective(objectives_func_list=self.objective_func, **self.states)
        if self.constrain_func is not None:
            self.constrains = func.MultiConstrain(self.constrain_func, self.states)
        else:
            self.constrains = None
        self.muteRate = self.uniprng.uniform(0.1,0.9) #Use "normalized" sigma
        self.numObj = len(self.objectives)
        self.frontRank = None
        self.crowdDist = None
        
    def mutateMutRate(self):
        self.muteRate=self.muteRate*math.exp(self.learningRate*self.uniprng.normalvariate(0,1))
        if self.muteRate < self.minMuteRate: self.muteRate=self.minMuteRate
        if self.muteRate > self.maxMuteRate: self.muteRate=self.maxMuteRate
             
    def dominates(self, other, compare_list = ["max", "max"]):

        if self is other:
            return 0
        
        if self.constrains is not None and other.constrains is not None:
            selfConstrain = np.sum(self.constrains.results == False)
            otherConstrain = np.sum(other.constrains.results == False)
            
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
    def __init__(self, img, pts, model):
        states = []
        self.theta_array = []
        self.img = img
        self.pts = pts
        self.pathLine = []
        self.model = model

        for i in range(len(pts) - 1):
            line = shape.Line(pts[i], pts[i + 1])
            self.pathLine.append(line)
        self.pathLine.append(shape.Line(pts[-1], pts[0]))

        lastPointOut = 0  
        for i in range(len(self.pts) - 1):
            pointIn = self.uniprng.uniform(lastPointOut, 1)
            theta = model.findTheta(self.pathLine[i].percentage2point(pointIn), self.pathLine[i + 1].a, self.pathLine[i + 1].b, self.pathLine[i + 1].c,
                                self.pathLine[i].theta, self.pathLine[i + 1].theta)
            if theta is not None:
                lastPointOut = model.calEndXY(self.pathLine[i].percentage2point(pointIn), theta, self.pathLine[i].theta, self.pathLine[i+1].theta)
                lastPointOut = np.linalg.norm(lastPointOut - pts[i+1]) / self.pathLine[i+1].length
            else:
                lastPointOut = 0
                
            if lastPointOut < 0 or lastPointOut >= 1:
                lastPointOut = 0
                theta = None
            states.append([pointIn, lastPointOut])
            self.theta_array.append(theta)
        pointIn = self.uniprng.uniform(lastPointOut, 1)
        theta = model.findTheta(self.pathLine[-1].percentage2point(pointIn), self.pathLine[0].a, self.pathLine[0].b, self.pathLine[0].c,
                            self.pathLine[-1].theta, self.pathLine[0].theta)
        if theta is not None:
            lastPointOut = model.calEndXY(self.pathLine[-1].percentage2point(pointIn), theta, self.pathLine[-1].theta, self.pathLine[0].theta)
            lastPointOut = np.linalg.norm(lastPointOut - pts[0]) / self.pathLine[0].length
        else:
            lastPointOut = 0 
        if lastPointOut > states[0][0] or lastPointOut < 0 or lastPointOut >= 1:
            lastPointOut = 0
            theta = None
        states.append([pointIn, lastPointOut])
        self.theta_array.append(theta)

        self.states = {"states": states, "theta_array": self.theta_array, "line": self.pathLine, "dilate_radius": self.model.d, "model": self.model}
        obstacleConstrain = func.constrain.ObstacleCollision(self.img)
        super().__init__(objective_func=[func.fitness.smoothCurveFitness, func.fitness.straightLineFitness], constrain_func=[obstacleConstrain, func.constrain.constModelConstrain])
    def crossover(self, other):
        self.constrains.results
        other.constrains.results
    def mutate(self):
        self.mutateMutRate() #update mutation rate
        
        '''
        It's your choice to use any mutate function you learned from our course. 
        '''

        for i in range(len(self.state)):
            
            # if self.uniprng.random() < self.mutRate:
            self.state[i] = self.state[i] + self.mutRate*self.normprng.normalvariate(0,1)
            self.state[i] = int( self.__constrain( round( self.state[i] ), 0, self.nSpells-1 ) )
        self.objectives=None