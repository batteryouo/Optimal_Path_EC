#
# Population.py
#
#

import copy
import math
from operator import attrgetter
from . import individual
import matplotlib.pyplot as plt

class PathPopulation:
    """
    Population
    """
    crossoverFraction=None
    
    def __init__(self, img, pts, model, populationSize, uniprng, normprng):
        """
        Population constructor
        """
        self.population=[]
        for _ in range(populationSize):
            self.population.append(individual.PathIndividual(img, pts, model, uniprng, normprng))
        self.uniprng = uniprng
        self.normprng = normprng
        self.updateRanking()                                                                                                                               

    def __len__(self):
        return len(self.population)
    
    def __getitem__(self,key):
        return self.population[key]
    
    def __setitem__(self,key,newValue):
        self.population[key]=newValue
        
    def copy(self):
        return copy.deepcopy(self)

    def evaluateObjectives(self):
        for ind in self.population: 
            ind.evaluateObjectives()
            
    def mutate(self):     
        for individual in self.population:
            individual.mutate()
            
    def crossover(self):
        indexList1=list(range(len(self)))
        indexList2=list(range(len(self)))
        self.uniprng.shuffle(indexList1)
        self.uniprng.shuffle(indexList2)
            
        if self.crossoverFraction == 1.0:             
            for index1,index2 in zip(indexList1,indexList2):
                self[index1].crossover(self[index2])
        else:
            for index1,index2 in zip(indexList1,indexList2):
                rn=self.uniprng.random()
                if rn < self.crossoverFraction:
                    # self[index1].complementCrossover(self[index2])     
                    self[index1].floatCrossover(self[index2])     

    def combinePops(self,otherPop):
        self.population.extend(otherPop.population)

    def computeCrowding(self):
        """
        Compute crowding metric using k-th nearest-neighbor w/ normalized distance.
        """
        
        if len(self.population) == 0: return #nothing to do
        
        # if single objective, set all densities to zero then return
        if self.population[0].numObj == 1:
            for ind in self.population:
                ind.crowdDist=0.0
            return
                
        # compute k for knn density estimate
        kdist=int(math.sqrt(len(self.population)))
        
        # compute normalization vector
        maxObj=self.population[0].objectives.copy()
        minObj=self.population[0].objectives.copy()
        for ind in self.population:
            for i in range(ind.numObj):
                if ind.objectives[i] < minObj[i]: minObj[i]=ind.objectives[i]
                if ind.objectives[i] > maxObj[i]: maxObj[i]=ind.objectives[i]
        
        normVec=[]        
        for min,max in zip(minObj,maxObj):
            norm=math.fabs(max-min)
            if norm == 0: norm=1.0 #watch out for possible divide by zero problems
            normVec.append(norm)    
        
        # init distance matrix
        distanceMatrix=[]
        for i in range(len(self.population)):
            distanceMatrix.append([0.0]*len(self.population))
        
        # compute distance matrix
        # (matrix is diagonally symmetric so only need to compute half, then reflect)
        for i in range(len(self.population)):
            for j in range(i+1):
                distanceMatrix[i][j]=self.population[i].distance(self.population[j],normVec)
                distanceMatrix[j][i]=distanceMatrix[i][j]
                      
        # sort the rows by distance
        for row in distanceMatrix:
            row.sort()
        
        # find the crowding distance using knn index
        i=0
        for ind in self.population:
            ind.crowdDist=distanceMatrix[i][kdist]
            i+=1

    def computeFrontRanks(self):
        """
        Compute non-dominated front ranks using NSGA-II front-ranking scheme
        It's your job to implement this function.
        You could utilize the function from your (or the answer's) HW7!
        """

        tmpPop = copy.deepcopy(self.population)
        self.population.clear()
        rank = 0
        while len(tmpPop) > 0:
            currentFront = []
            for ind in tmpPop:
                dominated = False
                for otherInd in tmpPop:
                    if ind.dominates(otherInd) == -1:
                        dominated = True
                        break

                if not dominated:
                    currentFront.append(ind)

            for ind in currentFront:
                ind.frontRank = rank
                self.population.append(ind)
                tmpPop.remove(ind)
            rank += 1
       

    def binaryTournament(self):
        """
        It's your job to implement this function.
        You could utilize the function from your (or the answer's) HW7!
        """
        
        newPop = []
        for _ in range(2):
            tournamentPairIndex = [ind for ind in range(len(self.population))]
            # pop is a list or some objects that is mutable, the shuffle methods will mess up the order of original data 
            self.uniprng.shuffle(tournamentPairIndex)

            for i in range(len(tournamentPairIndex) // 2):
                
                player = [ self.population[tournamentPairIndex[i]], self.population[tournamentPairIndex[i + 1]]]
                
                comparison = player[0].compareRankAndCrowding(player[1])
                if comparison == 1:
                    winner = player[0]
                elif comparison == -1:
                    winner = player[1]
                else:
                    winner = self.uniprng.choice(player)
                newPop.append(winner)
        
        # overwrite old pop with newPop (i.e., the selected pop)   
        self.population=newPop    
    def MOTruncation(self,newpopsize):
        
        '''
        It's your job to implement this function.
        You need to build the truncation function that based on non-dominated front ranking & Knn crowding.
        '''

        # sort by front rank and crowding distance
        self.population.sort(key=lambda ind: (ind.frontRank, -ind.crowdDist))
        #then truncate the bottom
        self.population=self.population[:newpopsize]

        
    def updateRanking(self):
        """
        Update front-rank and crowding distance for entire population
        """
        self.computeFrontRanks()
        self.computeCrowding()
    
    # def generatePlots(self,title=None,showPlot=True):
    #     '''
    #     It's your job to implement this function.
    #     You could utilize the function from your (or the answer's) HW7!
    #     '''
    #     #first, make sure state & objective space have at least 2 dimensions, pop size at least 1
    #     if len(self.population) < 1:
    #         raise Exception('showPlots error: Population size must be >= 1 !')
    #     if (len(self.population[0].state) < 2) or (len(self.population[0].objectives) < 2):
    #         raise Exception('showPlots error: State & objective spaces must have at least 2 dimensions!')
    
    #     #if front ranking has not been computed, then skip
    #     # the front-rank plot
    #     if self.population[0].frontRank is None: plotOrder=[121,122,000]
    #     else: plotOrder=[121,122]

    #     #top-level attributes for collection of subplots
    #     if title is not None:
    #         fig, axs = plt.subplots(13)
    #         fig.suptitle(title)
    #     plt.subplots_adjust(wspace=0.75) #increase spacing between plots a bit
        
    #     # #individuals in state space
    #     # plt.subplot(plotOrder[0])
    #     # x=[ind.state[0] for ind in self.population]
    #     # y=[ind.state[1] for ind in self.population]
    #     # plt.scatter(x,y)
    #     # plt.xlabel('x1')
    #     # plt.ylabel('x2')
    #     # plt.title('State Space')
        
    #     #individuals in objective space
    #     plt.subplot(plotOrder[0])
    #     x=[ind.objectives[0] for ind in self.population]
    #     y=[ind.objectives[1] for ind in self.population]
    #     plt.scatter(x,y)
    #     plt.xlabel('f1')
    #     plt.ylabel('f2')
    #     plt.title('Objective Space')
        
        
    #     #Note: If front ranks have not been computed, then
    #     #      skip the frontRank plot...
    #     if self.population[0].frontRank is not None:
    #         #non-dominated ranked fronts in objective space
    #         plt.subplot(plotOrder[1])   
            
    #         #first, let's find highest front rank
    #         maxRank=0
    #         for ind in self.population:
    #             if ind.frontRank > maxRank: maxRank=ind.frontRank
            
    #         rank=0
    #         while rank <= maxRank:
    #             xy=[ind.objectives for ind in self.population if ind.frontRank == rank]
    #             xy.sort(key=lambda obj: obj[0]) #need to sort in 1st dim to make connected line plots look sensible!
    #             x=[obj[0] for obj in xy]
    #             y=[obj[1] for obj in xy]
    #             plt.plot(x,y,marker='o',label=str(rank))
    #             rank+=1
                
    #         plt.xlabel('f1')
    #         plt.ylabel('f2')
    #         plt.title('Ranked Fronts')
    #     if showPlot:
    #         plt.show()
        
                
    def __str__(self):
        s=''
        for ind in self:
            s+=str(ind) + '\n'
        return s


        
       
