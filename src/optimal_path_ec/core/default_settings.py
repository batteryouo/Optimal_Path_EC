import numpy as np

from . import ec

def aa(states):
    return 1


ec.Individual.uniprng = np.random.default_rng(123)
ec.Individual.normprng = np.random.default_rng(456)
ec.Individual.objective_func = [aa, aa]
ec.Individual.constrain_func = None