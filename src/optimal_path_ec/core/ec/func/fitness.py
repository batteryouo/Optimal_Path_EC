import numpy as np

def smoothCurveFitness(theta_array):
    counter = 0
    theta_square = 0
    for theta in theta_array:
        if theta is None:
            continue
        theta_square += theta**2
        counter += 1
    if counter == 0:
        return 0
    
    return np.sqrt(theta_square) / counter
            
def straightLineFitness(states, line):
    LineLength_squareSum = 0
    for i in range(len(states)):
        LineLength_squareSum += np.linalg.norm(line[i].percentage2point(states[i][0]) - line[i].percentage2point(states[i-1][1]) ) ** 2
        
    return np.sqrt(LineLength_squareSum) / len(states)