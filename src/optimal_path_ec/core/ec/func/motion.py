import numpy as np

class ConstMotion():
    d = 1 
    def __init__(self, d):
        self.d = d

    def calW(self, V, theta, d = None):
        if self.d is None:
            self.d = d
        return V/self.d * np.sin(theta)
    
    @classmethod
    def calToward(cls, initToward, w, timeStep):
        return initToward + w*(timeStep[1] - timeStep[0])

    @classmethod
    def calXY(cls, initPos, initToward, V, timeStep , w):
        x = initPos[0] + V/w(np.sin(initToward + w*timeStep) - np.sin(initToward + w*0))
        y = initPos[1] - V/w(np.cos(initToward + w*timeStep) - np.cos(initToward + w*0))

        return np.array([x, y])
    
    def findTheta(self, pt1, a1, b1, c1, initToward, finalToward):
        # print(f"pt1: {pt1}, a1: {a1}, b1: {b1}, c1: {c1}, initToward: {initToward}, finalToward: {finalToward}" )
        if initToward == finalToward:
            return 0
        h0 = np.sin(finalToward) - np.sin(initToward)
        h1 = np.cos(initToward) - np.cos(finalToward)
        
        distanceOfline = a1*pt1[0] + b1*pt1[1] + c1
        turnFactor = self.d*(a1*h0 + b1*h1)
        # print(f"distanceOfline: {distanceOfline}, turnFactor: {turnFactor}, pt: {pt1}")
        if abs(distanceOfline / turnFactor) > 1:
            return None           
        
        return np.arcsin(-distanceOfline/turnFactor)

class Motion():
    d = 1
    def __init__(self, d):
        self.d = d

    @classmethod
    def calW(cls, V, theta, d = None):

        if cls.d is None:
            cls.d = d

        return V/cls.d * np.sin(theta)
    
    @classmethod
    def calToward(cls, initToward, w, timeSequence):
        if not isinstance(w, np.ndarray):
            try:
                w = np.array(w, dtype=float)
            except Exception:
                raise TypeError("Theta parameter must be a sequence convertible to an array.")

        if not isinstance(timeSequence, np.ndarray):
            try:
                timeSequence = np.array(timeSequence, dtype=float)
            except Exception:
                raise TypeError("timeSequence parameter must be a sequence convertible to an array.")

        if w.size == 0 or timeSequence.size == 0:
             raise ValueError("w and timeSequence arrays must not be empty.")
        if w.size != timeSequence.size:
             raise ValueError("w array and timeSequence array must have the same length.")
        if np.any(timeSequence <= 0):
             raise ValueError("All elements in timeSequence must be positive numbers.")
         
        delta_theta = w * timeSequence
        cumulative_theta_change = np.cumsum(delta_theta)
        toward_sequence = np.insert(cumulative_theta_change, 0, 0.0) + initToward

        return toward_sequence[-1]

    @classmethod
    def calXY(cls, initPos, initToward, V, vTimeSequence , w, wTimeSequence):
        if not isinstance(V, np.ndarray):
            V = np.array(V, dtype=float)
        if not isinstance(vTimeSequence, np.ndarray):
            vTimeSequence = np.array(vTimeSequence, dtype=float)
        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=float)
        if not isinstance(wTimeSequence, np.ndarray):
            wTimeSequence = np.array(wTimeSequence, dtype=float)

        # Check size constraints
        if V.size == 0 or vTimeSequence.size == 0:
            raise ValueError("V and vTimeSequence arrays must not be empty.")
        if w.size == 0 or wTimeSequence.size == 0:
            raise ValueError("w and wTimeSequence arrays must not be empty.")
        if V.size != vTimeSequence.size:
            raise ValueError("V array and vTimeSequence array must have the same length.")
        if w.size != wTimeSequence.size:
            raise ValueError("w array and wTimeSequence array must have the same length.")
        total_time_v = np.sum(vTimeSequence)
        total_time_w = np.sum(wTimeSequence)
        
        # Check if the total duration of V and W sequences are equal within a small tolerance
        if not np.isclose(total_time_v, total_time_w):
             raise ValueError(
                f"Total time duration mismatch: Sum of vTimeSequence ({total_time_v:.6f}) "
                f"must equal Sum of wTimeSequence ({total_time_w:.6f})."
            )
        v_edges = np.concatenate([[0], np.cumsum(vTimeSequence)])
        w_edges = np.concatenate([[0], np.cumsum(wTimeSequence)])
        unified_edges = np.unique(np.concatenate([v_edges, w_edges]))
        
        v_idx = 0
        w_idx = 0
        x, y = initPos
        theta = initToward
        xs = [x]
        ys = [y]
        thetas = [theta]
        for i in range(len(unified_edges)-1):
            t0 = unified_edges[i]
            t1 = unified_edges[i+1]
            dt = t1 - t0
            # update v_idx
            while v_idx+1 < len(v_edges) and t0 >= v_edges[v_idx+1]:
                v_idx += 1
            
            # update w_idx
            while w_idx+1 < len(w_edges) and t0 >= w_edges[w_idx+1]:
                w_idx += 1
            v_now = V[v_idx]
            w_now = w[w_idx]
            if w_now != 0:
                x += v_now/w_now*(np.sin(theta + w_now*dt) - np.sin(theta + w_now*0))
                y += -v_now/w_now*(np.cos(theta + w_now*dt) - np.cos(theta + w_now*0))
            else:
                x += v_now*np.cos(theta)*dt
                y += v_now*np.sin(theta)*dt
            theta = cls.calToward(theta, w_now, dt)
            xs.append(x)
            ys.append(y)
            thetas.append(theta)

        return xs, ys
