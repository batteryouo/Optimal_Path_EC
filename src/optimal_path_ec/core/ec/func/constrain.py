import inspect

import cv2
import numpy as np

from . import motion

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

class MultiConstrain():
    
    def __init__(self, func_list:list, states):
        self.constrain_funcs = [] 
        self.results = []

        for func in func_list:
            sig = inspect.signature(func)

            param_names = set(sig.parameters.keys())
            
            has_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            
            self.constrain_funcs.append((func, param_names, has_varkw))
        for func, param_names, has_varkw in self.constrain_funcs:
            if has_varkw:
                func_kwargs = states 
            else:
                common_keys = param_names.intersection(states.keys())
                func_kwargs = {k: states[k] for k in common_keys}
            
            self.results.append(func(**func_kwargs))
        self.results = np.array(flatten(self.results))  
    def __call__(self, **kwargs):
        self.results = []

        for func, param_names, has_varkw in self.constrain_funcs:
            if has_varkw:
                func_kwargs = kwargs
            else:
                common_keys = param_names.intersection(kwargs.keys())
                func_kwargs = {k: kwargs[k] for k in common_keys}
            
            self.results.append(func(**func_kwargs))
        self.results = np.array(np.array(flatten(self.results)))  
        return self.results
    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]
    
class ObstacleCollision():
    
    def __init__(self, img):
        if len(img.shape) == 3:
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 4:
            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        self.img = img
        
    def isSafe(self, pts, dilate_radius:int = 1):
        """Check if pass through the obstacle\n
        #### 0: safe region
        #### otherwise: obstacle

        Args:
            pts (np.ndarray): __description
            dilate_radius (int, optional): _description_. Defaults to 1.

        Returns:
            bool: _description_
        """
        h, w = self.img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(pts) - 1):
            pt1 = self.boundingPoint(pts[i])
            pt1 = int(pt1[1]), int(pt1[0])
            pt2 = self.boundingPoint(pts[i + 1])
            pt2 = int(pt2[1]), int(pt2[0])
            
            cv2.line(mask, pt1, pt2, 255, dilate_radius, cv2.LINE_AA)
            # cv2.line(self.img, pt1, pt2, 125, dilate_radius, cv2.LINE_AA)
        collision_roi = cv2.bitwise_and(self.img, mask)
        pixel_count = cv2.countNonZero(collision_roi)
        # cv2.namedWindow("i", 0)
        # cv2.imshow("i", self.img)
        # cv2.waitKey(0)
        if pixel_count > 0:
            return False
        else:
            return True
         
    def boundingPoint(self, pt):
        if pt[0] < 0:
            pt[0] = 0
        if pt[0] > self.img.shape[0]:
            pt[0] = self.img.shape[0] - 1
        
        if pt[1] < 0:
            pt[1] = 0
        if pt[1] > self.img.shape[1]:
            pt[1] = self.img.shape[1] -1
        
        return pt 
    
    def __call__(self, model, states, theta_array, line, dilate_radius:int = 1):
        self.results = []
        v = 1
        dt = 0.1
        for i in range(len(states)):
            if i >= len(states) - 1:
                i = -1
            if theta_array[i] is None:
                self.results.append(False)
                continue
            theta = theta_array[i]
            w = model.calW(1, theta)
            toward = line[i].theta
            initial_point = line[i].percentage2point(states[i][0])
            target_point = line[i + 1].percentage2point(states[i][1])
            pt = initial_point
            pts = [pt]
            while np.linalg.norm(pt - target_point) > 1:
                pt = model.calXY(pt, toward, v, dt, w)
                toward = model.calToward(toward, w, dt)
                pts.append(pt)
            self.results.append(self.isSafe(pts, dilate_radius))  
        
        return self.results
        

def constModelConstrain(theta_array):
    modelConstrain = [True if theta is not None else False for theta in theta_array]

    return modelConstrain