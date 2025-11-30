import numpy as np

class ConstMotion():
    d = None
    def __init__(self, d):
        self.d = d

    @classmethod
    def calW(cls, V, theta, d = None):
        if cls.d is None:
            cls.d = d
        return V/cls.d * np.sin(theta)
    
    @classmethod
    def calToward(cls, initToward, w, timeStep):
        return initToward + w*(timeStep[1] - timeStep[0])

class Motion():
    d = 1
    def __init__(self, d):
        self.d = d

    @classmethod
    def calW(cls, V, vTimeStep, theta, thetaTimeStep, d = None):

        if cls.d is None:
            cls.d = d

        raise NotImplementedError
    
    @classmethod
    def calToward(cls, initToward, w, timeStep):
        raise NotImplementedError