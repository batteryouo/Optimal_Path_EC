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

def main():
    mapGenerator = map_generator.Generator(333)
    mapGenerator.createFramework((500, 500), 10)
    pts = None
    img = None
    while pts is None:
        pts, img = mapGenerator.generate(20, 100, 400, 30, 40)
    color_map = mapGenerator.drawLineAndPoints(img, pts)
    cv2.namedWindow("canvas", 0)
    cv2.imshow("canvas", color_map)
    
    model = core.ec.ConstMotion(10)
    core.ec.PathIndividual(img, pts, model, uniprng, normprng)
    cv2.waitKey(0)
    population=core.ec.PathPopulation(img ,pts, model, 10, uniprng, normprng)
    core.ec.PathPopulation.crossoverFraction = 0.8
    population.evaluateObjectives()
    for i in range(2):
        
        offspring=population.copy()
        offspring.binaryTournament()
        offspring.crossover()
        offspring.mutate()

        population.combinePops(offspring)
        population.evaluateObjectives()
        population.updateRanking()
        population.MOTruncation(10)
        # printStats(population,i+1)

if __name__ == "__main__":
    main()