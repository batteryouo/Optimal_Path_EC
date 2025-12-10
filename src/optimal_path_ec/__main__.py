import os

import cv2
import numpy as np

import core
import map_generator
# from utils import readYaml

# cfg = readYaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
# map_cfg = cfg["map"]

uniprng = np.random.default_rng(123)
normprng = np.random.default_rng(456)

def printStats(pop,gen):
    print('Generation:',gen)
    avgDamage=0
    avgCost=0
    costval,maxval=pop[0].objectives
    muteRate=pop[0].muteRate
    for ind in pop:
        avgDamage+=ind.objectives[1]
        avgCost+=ind.objectives[0]
        if ind.objectives[1] > maxval:
            costval,maxval=ind.objectives
            muteRate=ind.muteRate
        print(ind, f"frontRank: {ind.frontRank}", f"line: {ind.objectives[1]}", 
              f"curve: {ind.objectives[0]}, constrain: {ind.constrains.results}")

    print('')

def main():
    mapGenerator = map_generator.Generator(333)
    mapGenerator.createFramework((500, 500), 10)
    pts = None
    img = None
    while pts is None:
        pts, img = mapGenerator.generate(20, 100, 400, 30, 20)
    color_map = mapGenerator.drawLineAndPoints(img, pts)
    cv2.namedWindow("canvas", 0)
    cv2.imshow("canvas", color_map)
    
    model = core.ec.ConstMotion(10)
    core.ec.PathIndividual(img, pts, model, uniprng, normprng)
    cv2.waitKey(0)
    population=core.ec.PathPopulation(img ,pts, model, 30, uniprng, normprng)
    core.ec.PathPopulation.crossoverFraction = 1.0
    population.evaluateObjectives()
    for i in range(30):
        
        offspring1 = population.copy()
        offspring2 = population.copy()
        offspring3 = population.copy()

        offspring1.binaryTournament()  
        offspring1.floatCrossover()
        offspring1.mutate()
        population.combinePops(offspring1)

        offspring2.binaryTournament()
        offspring2.complementCrossover()
        offspring2.mutate()
        population.combinePops(offspring2)  

        offspring3.binaryTournament()
        offspring3.doublePointCrossover()
        offspring3.mutate()
        population.combinePops(offspring3)


        population.evaluateObjectives()
        population.updateRanking()
        population.MOTruncation(30)
        printStats(population,i+1)

if __name__ == "__main__":
    main()