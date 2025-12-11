import os

import cv2
import numpy as np

import core
import map_generator
import simulation
from utils import readJson

cfg = readJson(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", "config.json"))
map_cfg = cfg["map"]
random_seed_cfg = cfg["random_seed"]
model_cfg = cfg["model"]
ec_cfg = cfg["ec"]

uniprng = np.random.default_rng(random_seed_cfg["uniform"])
normprng = np.random.default_rng(random_seed_cfg["norm"])

def printStats(pop,gen):
    print('Generation:',gen)
    for ind in pop:
        print(ind, f"frontRank: {ind.frontRank}", f"line: {ind.objectives[1]}", 
              f"curve: {ind.objectives[0]}, constrain: {ind.constrains.results}")

    print('')

def main():
    mapGenerator = map_generator.Generator(random_seed_cfg["map_seed"])
    mapGenerator.createFramework(map_cfg["frameSize"], map_cfg["borderWidth"])
    pts = None
    img = None
    while pts is None:
        pts, img = mapGenerator.generate(map_cfg["max_points"], map_cfg["min_dist"], map_cfg["max_ranges"], map_cfg["min_angle_degrees"], map_cfg["expand_width"])
    color_map = mapGenerator.drawLineAndPoints(img, pts)
    cv2.namedWindow("canvas", 0)
    cv2.imshow("canvas", color_map)
    cv2.waitKey(0)
    model = core.ec.ConstMotion(model_cfg["d"])
    population=core.ec.PathPopulation(img ,pts, model, ec_cfg["population_size"], uniprng, normprng)
    core.ec.PathPopulation.crossoverFraction = ec_cfg["crossoverFraction"]
    population.evaluateObjectives()
    
    for i in range(ec_cfg["generation"]):
        
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
        population.MOTruncation(ec_cfg["population_size"])
        printStats(population,i+1)
    winner:core.ec.PathIndividual | None = None
    for ind in population:
        if ind.frontRank == 0:
            winner = ind
            break
    sim = simulation.Simulation(model, winner.states["states"], winner.states["theta_array"], winner.pathLine, color_map)
    print("press `esc` to quit")
    sim.run()
if __name__ == "__main__":
    main()