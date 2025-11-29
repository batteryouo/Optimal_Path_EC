import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import Random

N_SKELETON_POINTS = 130
MIN_DISTANCE = 500
MAX_RANGE = 2000
MIN_ANGLE_DEGREES = 40

class Generator():
    def __init__(self, seed = None):
        self.frameSize = (1000, 1000)
        self.borderWidth = 10
        self.mapSize = (980, 980)
        self.canvas = None
        if not isinstance(seed, int) or (seed is None):
            seed = 1234
        self.rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)
        
    def createFramework(self, frameSize = (1000, 1000), borderWidth:int = 10):

        if frameSize[0] < borderWidth*2 or frameSize[1] < borderWidth *2:
            raise ValueError("FrameSize is lower than two times of the borderWidth.")

        self.frameSize = frameSize
        self.borderWidth = borderWidth
        self.mapSize = (frameSize[0] - 2*borderWidth, frameSize[1] - 2*borderWidth)
        self.canvas = np.zeros(frameSize, dtype=np.uint8)
        self.canvas[borderWidth:-1*borderWidth, borderWidth:-1*borderWidth] = 255
    
    def generate(self, n:int, min_dist:float, max_range:float=None, min_angle_degrees:float=10):
        if max_range is None:
            max_range = np.min(self.mapSize)
        
        skeleton_points = self.generate_constrained_points(n, min_dist, max_range)
        ordered_skeleton = self.__connect_via_angular_sort(skeleton_points)
        optimized_skeleton = self.optimize_path_skeleton(ordered_skeleton, MIN_ANGLE_DEGREES)
        if(len(optimized_skeleton) >= 3):
            return None
            
        return optimized_skeleton
    
    def plot(self):
        pass

    def generate_constrained_points(self, n: int, min_dist: float, max_range: float = 200) -> np.ndarray:
        
        points = []
        max_attempts = n * 100
        
        for _ in range(max_attempts):
            if len(points) >= n:
                break

            candidate = np.random.rand(2) * max_range - (max_range / 2)
            
            is_too_close = False
            for existing_point in points:
                dist_sq = np.sum((candidate - existing_point)**2)
                if dist_sq < min_dist**2:
                    is_too_close = True
                    break
            
            if not is_too_close:
                points.append(candidate)
                
        return np.array(points)

    def __connect_via_angular_sort(self, points: np.ndarray) -> np.ndarray:

        if len(points) < 3:
            return points

        center = np.mean(points, axis=0)
        
        vectors = points - center
        
        angles = np.arctan2(vectors[:, 1], vectors[:, 0]) 
        
        sorted_indices = np.argsort(angles)
        
        sorted_path_skeleton = points[sorted_indices]
        
        return sorted_path_skeleton

    def __get_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculates the inner angle (in radians) formed by three points p1-p2-p3, 
        with p2 as the vertex.
        
        The result is always in the range [0, pi] radians (0 to 180 degrees).

        @param p1: Coordinates of the first point. 
        :type p1: numpy.ndarray
        @param p2: Coordinates of the angle's vertex. This point is subtracted to create the two vectors.
        :type p2: numpy.ndarray
        @param p3: Coordinates of the third point.
        :type p3: numpy.ndarray
        @returns: The angle value in radians.
        :rtype: float
        """
        vec1 = p1 - p2
        vec2 = p3 - p2
        
        dot_product = np.dot(vec1, vec2)
        
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return np.pi 
            
        # cos(theta) = (vec1 dot vec2) / (|vec1| * |vec2|)
        cosine_angle = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)
        
        return np.arccos(cosine_angle)

    def optimize_path_skeleton(self, skeleton: np.ndarray, min_angle_degrees: float) -> np.ndarray:
        """
        Optimizes a closed path skeleton by removing interior points if the angle 
        formed by three consecutive points is smaller than a specified threshold.

        @param skeleton: The angle-sorted closed skeleton points.
        :type skeleton: numpy.ndarray
        @param min_angle_degrees: The minimum allowed angle (in degrees).
        :type min_angle_degrees: float
        @returns: The optimized set of new skeleton points.
        :rtype: numpy.ndarray
        """
        if len(skeleton) < 4:
            return skeleton

        min_angle_radians = np.deg2rad(min_angle_degrees)
        
        # [P_n-1, P_0, P_1, ..., P_n-1, P_0]
        num_pts = len(skeleton)
        current_skeleton = list(skeleton)
        
        optimized = False

        while not optimized:
            optimized = True
            
            n = len(current_skeleton)
            
            if n < 4:
                break
            i = 0
            while i < n:
                
                p_prev = current_skeleton[(i - 1 + n) % n]
                p_curr = current_skeleton[i]
                p_next = current_skeleton[(i + 1) % n]
                
                angle = self.__get_angle(p_prev, p_curr, p_next)
                
                if angle < min_angle_radians:
                    current_skeleton.pop(i)
                    n -= 1 
                    optimized = False
                else:
                    i += 1
                    
            if not optimized:
                i = 0
        if len(current_skeleton) < num_pts:
            logging.info("Only {len(current_skeleton)}/{num_pts} points met the distance constraint.")

        return np.array(current_skeleton)
