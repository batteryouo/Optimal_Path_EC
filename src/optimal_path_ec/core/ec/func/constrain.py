import inspect

import cv2
import numpy as np

from . import motion

class MultiConstrain():
    
    def __init__(self, func_list, state):
        self.constrain_funcs = func_list
        self.results = []
        for param in state:
            self.results.append(self.constrain_funcs(param))
        
    def __call__(self, state):
        self.results = []
        for param in state:
            self.results.append(self.constrain_funcs(param))
            
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
        if len(self.shape) == 4:
            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
        img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
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

        collision_roi = cv2.bitwise_and(self.img, mask)
        pixel_count = cv2.countNonZero(collision_roi)
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
    
    def __call__(self, pts, dilate_radius:int = 1):
        return self.isSafe(pts, dilate_radius)

def constModelConstrain(pt1, a1, b1, c1, d, initToward, finalToward):
    model = motion.ConstMotion(d)
    theta = model.findTheta(pt1, a1, b1, c1,  d, initToward, finalToward)
    
    if theta is None:
        return False
    
    return True