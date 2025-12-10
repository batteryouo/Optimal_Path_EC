import copy 
import math

import numpy as np

from . import objective
from . import func
from . import shape
from . func import motion

class Individual():


    def __init__(self, objective_func = None, constrain_func = None, minMutRate = 1e-100, maxMutRate = 1, learningRate = 1e-1, uniprng = None, normprng = None):
        self.objective_func = objective_func
        
        self.objective_func = objective_func
        self.constrain_func = constrain_func
        self.minMuteRate = minMutRate
        self.maxMuteRate = maxMutRate
        self.learningRate = learningRate
        self.uniprng = uniprng
        self.normprng = normprng      
        self.objectives = objective.MultiObjective(objectives_func_list=self.objective_func, **self.states)
        if self.constrain_func is not None:
            self.constrains = func.MultiConstrain(self.constrain_func, self.states)
        else:
            self.constrains = None
        self.muteRate = self.uniprng.uniform(0.1,0.9) #Use "normalized" sigma
        self.numObj = len(self.objectives)
        self.frontRank = None
        self.crowdDist = None
    def evaluateObjectives(self):
        self.objectives(**self.states)
        if self.constrains is not None:
            self.constrains(**self.states)
            
    def mutateMuteRate(self):
        self.muteRate=self.muteRate*math.exp(self.learningRate*self.uniprng.normal(0,1))
        if self.muteRate < self.minMuteRate: self.muteRate=self.minMuteRate
        if self.muteRate > self.maxMuteRate: self.muteRate=self.maxMuteRate
             
    def dominates(self, other, compare_list = ["min", "max"]):

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
    def __init__(self, img, pts, model, uniprng, normprng):
        states = []
        self.theta_array = []
        self.img = img
        self.pts = pts
        self.pathLine = []
        self.model = model
        self.uniprng = uniprng
        self.normprng = normprng

        for i in range(len(pts) - 1):
            line = shape.Line(pts[i], pts[i + 1])
            self.pathLine.append(line)
        self.pathLine.append(shape.Line(pts[-1], pts[0]))

        lastPointOut = 0  
        for i in range(len(self.pts)):
            ind = i
            if ind >= len(self.pts)-1:
                ind = -1
                theta, pointIn, lastPointOut = self.generateSingleState(lastPointOut, self.pathLine[ind], self.pathLine[ind+1], pts[ind+1], states[ind+1][0])
            else:
                theta, pointIn, lastPointOut = self.generateSingleState(lastPointOut, self.pathLine[ind], self.pathLine[ind+1], pts[ind+1])
            states.append([pointIn, lastPointOut])
            self.theta_array.append(theta)

        self.states = {"states": states, "theta_array": self.theta_array, "line": self.pathLine, "dilate_radius": self.model.d, "model": self.model}
        obstacleConstrain = func.constrain.ObstacleCollision(self.img)
        super().__init__(objective_func=[func.fitness.smoothCurveFitness, func.fitness.straightLineFitness], constrain_func=[obstacleConstrain, func.constrain.constModelConstrain],
                         uniprng=uniprng, normprng=normprng)
    
    def complementCrossover(self, other):
        selfStates = self.states["states"]
        otherStates = other.states["states"]
        selfConstrain = self.constrains.results
        otherConstrain = other.constrains.results

        for i in range(len(selfStates)):
            if (not selfConstrain[i] or not selfConstrain[i + len(selfStates)]) and (otherConstrain[i] and otherConstrain[i + len(selfStates)]):
                self.states["states"][i] = copy.deepcopy(otherStates[i])
                self.theta_array[i] = copy.deepcopy(other.theta_array[i])
            self.spliceCheck(i)
            if (not otherConstrain[i] or not otherConstrain[i + len(selfStates)]) and (selfConstrain[i] and selfConstrain[i + len(selfStates)]):
                other.states["states"][i] = copy.deepcopy(selfStates[i])
                other.theta_array[i] = copy.deepcopy(self.theta_array[i])
            other.spliceCheck(i)

        self.states["theta_array"] = self.theta_array
        other.states["theta_array"] = other.theta_array    
            
    def floatCrossover(self, other):
        selfStates = self.states["states"]
        otherStates = other.states["states"]
        for i in range(-1, len(selfStates)-1):
            alpha = self.uniprng.uniform(0, 1)
            selfPointIn = alpha*selfStates[i][0] + (1-alpha)*otherStates[i][0]
            otherPointIn = (1-alpha)*selfStates[i][0] + alpha*otherStates[i][0]
            selfTheta = self.model.findTheta(self.pathLine[i].percentage2point(selfPointIn), self.pathLine[i+1].a, self.pathLine[i+1].b, self.pathLine[i+1].c
                                             , self.pathLine[i].theta, self.pathLine[i+1].theta)
            otherTheta = other.model.findTheta(other.pathLine[i].percentage2point(otherPointIn), other.pathLine[i+1].a, other.pathLine[i+1].b, other.pathLine[i+1].c
                                             , other.pathLine[i].theta, other.pathLine[i+1].theta)
            if selfTheta is not None:
                lastPointOut = self.model.calEndXY(self.pathLine[i].percentage2point(selfPointIn), selfTheta, self.pathLine[i].theta, self.pathLine[i+1].theta)
                lastPointOut = np.linalg.norm(lastPointOut - self.pts[i+1]) / self.pathLine[i+1].length
            else:
                lastPointOut = 0
            if not self.generateCheck(lastPointOut, selfStates[i+1][0]):
                lastPointOut = 0
                selfTheta = None
            else:
                self.states["states"][i] = [selfPointIn, lastPointOut]
                self.theta_array[i] = selfTheta
            self.spliceCheck(i)
            
             
            if otherTheta is not None:
                lastPointOut = other.model.calEndXY(other.pathLine[i].percentage2point(otherPointIn), otherTheta, other.pathLine[i].theta, other.pathLine[i+1].theta)
                lastPointOut = np.linalg.norm(lastPointOut - other.pts[i+1]) / other.pathLine[i+1].length
            else:
                lastPointOut = 0
            if not other.generateCheck(lastPointOut, otherStates[i+1][0]):
                lastPointOut = 0
                otherTheta = None
            else:
                other.states["states"][i] = [otherPointIn, lastPointOut]
                other.theta_array[i] = otherTheta
            other.spliceCheck(i)           
        # self.clear()
        # other.clear() 
        self.states["theta_array"] = self.theta_array
        other.states["theta_array"] = other.theta_array
               
    def doublePointCrossover(self, other):
        selfStates = self.states["states"]
        otherStates = other.states["states"]
        selfTheta = self.states["theta_array"]
        otherTheta = other.states["theta_array"]
        assert len(selfStates) == len(otherStates), "Parents must be the same length."
        n = len(selfStates)
        if n < 3:
            p1 = 0
            p2 = 1
        else:
            p1, p2 = sorted(self.uniprng.choice(np.arange(1, n), size=2, replace=False))
        
        tmp = selfStates[:p1] + otherStates[p1:p2] + selfStates[p2:]
        otherStates = otherStates[:p1] + selfStates[p1:p2] + otherStates[p2:]
        selfStates = tmp

        tmp = selfTheta[:p1] + otherTheta[p1:p2] + selfTheta[p2:]
        otherTheta = otherTheta[:p1] + selfTheta[p1:p2] + otherTheta[p2:]
        selfTheta = tmp
        self.theta_array = selfTheta
        other.theta_array = otherTheta

        self.states = {"states": selfStates, "theta_array": self.theta_array, "line": self.pathLine, "dilate_radius": self.model.d, "model": self.model} 
        other.states = {"states": otherStates, "theta_array": other.theta_array, "line": other.pathLine, "dilate_radius": other.model.d, "model": other.model} 
        for i in range(len(selfStates)):
            self.spliceCheck(i)
            other.spliceCheck(i)         
        # self.clear()
        # other.clear() 
    
    def mutate(self):
        self.mutateMuteRate() #update mutation rate
        states = self.states["states"]
        for i in range(len(states)):
            ind = i
            if ind >= len(states) - 1:
                ind = -1
            if self.uniprng.uniform(0, 1) < self.muteRate:
                theta, pointIn, lastPointOut = self.generateSingleState(states[ind-1][1], self.pathLine[ind], self.pathLine[ind+1], self.pts[ind+1], states[ind+1][0])
                self.states["states"][i] = [pointIn, lastPointOut]
                self.theta_array[i] = theta
        
        self.states = {"states": states, "theta_array": self.theta_array, "line": self.pathLine, "dilate_radius": self.model.d, "model": self.model} 
        self.clear()
        
    def generateSingleState(self, lastPointOut, currentLine, nextLine, endOfCurLine, nextTurnInPoint = None):
        pointIn = self.uniprng.uniform(lastPointOut, 1)
        theta = self.model.findTheta(currentLine.percentage2point(pointIn), nextLine.a, nextLine.b, nextLine.c, currentLine.theta, nextLine.theta)
        if theta is not None:
            lastPointOut = self.model.calEndXY(currentLine.percentage2point(pointIn), theta, currentLine.theta, nextLine.theta)
            lastPointOut = np.linalg.norm(lastPointOut - endOfCurLine) / nextLine.length
        else:
            lastPointOut = 0
            
        if not self.generateCheck(lastPointOut, nextTurnInPoint):
            lastPointOut = 0
            theta = None 
        return theta, pointIn, lastPointOut
    
    def generateCheck(self, lastPointOut, nextTurnInPoint = None):    
        if lastPointOut <= 0 or lastPointOut >= 1:
            return False
        if nextTurnInPoint is not None:
            if lastPointOut > nextTurnInPoint:
                return False
        return True
    
    def spliceCheck(self, p):
        states = self.states["states"]
        if p == len(states) - 1:
            p = -1
        if states[p][0] < states[p-1][1]:
            self.states["states"][p] = [0, 0]
            self.states["theta_array"][p] = None
            self.theta_array[p] = None
            return False
        if states[p][1] > states[p+1][0]:
            self.states["states"][p] = [0, 0]
            self.states["theta_array"][p] = None
            self.theta_array[p] = None
            return False
        return True
    
    def clear(self):
        self.objectives.values=None
        self.constrains.results = None 