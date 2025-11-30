import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import Random

class Generator():
    frameSize = (1000, 1000)
    borderWidth = 10
    mapSize = (980, 980)
    canvas = None

    def __init__(self, seed = None):
        
        self.canvas = None
        if not isinstance(seed, int) or (seed is None):
            seed = 1234
        self.rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

        self.points = None
        
    def createFramework(self, frameSize = (1000, 1000), borderWidth:int = 10):

        if frameSize[0] < borderWidth*2 or frameSize[1] < borderWidth *2:
            raise ValueError("FrameSize is lower than two times of the borderWidth.")

        self.frameSize = frameSize
        self.borderWidth = borderWidth
        self.mapSize = (frameSize[0] - 2*borderWidth, frameSize[1] - 2*borderWidth)
        self.canvas = np.zeros(frameSize, dtype=np.uint8)
        self.canvas = np.full(frameSize, 0, dtype=np.uint8)
        self.canvas[borderWidth:frameSize[0]-1*borderWidth, borderWidth:frameSize[1]-1*borderWidth] = 255
    
    def generate(self, n:int, min_dist:float, max_range:float=None, min_angle_degrees:float=10, expand_width = 10):
        max_range -= expand_width
        if max_range is None:
            max_range = np.min(self.mapSize) - expand_width
        
        if max_range < 0:
            raise ValueError("max range is lower than expand_width")

        skeleton_points = self.generate_constrained_points(n, min_dist, max_range)
        ordered_skeleton = self.__connect_via_angular_sort(skeleton_points)
        optimized_skeleton = self.optimize_path_skeleton(ordered_skeleton, min_angle_degrees)
        if(len(optimized_skeleton) < 3):
            return (None, self.canvas)
        
        centerOfFrame = (self.frameSize[0]/2, self.frameSize[1]/2)
        optimized_skeleton += centerOfFrame

        self.points = optimized_skeleton
        self.canvas = self.draw_smooth_expand_track(self.points, expand_width)
        
        return optimized_skeleton, self.canvas
    
    def clear_canvas(self):
        self.canvas = np.zeros(self.frameSize, dtype=np.uint8)
        self.canvas = np.full(self.frameSize, 0, dtype=np.uint8)
        self.canvas[self.borderWidth:self.frameSize[0]-1*self.borderWidth, self.borderWidth:self.frameSize[1]-1*self.borderWidth] = 255 

    def draw_smooth_expand_track(self, pts, expand_width):

        height, width = self.canvas.shape

        img = np.zeros((height, width), dtype=np.uint8) + 255
        center_line_pts = np.array(pts, np.int32)[:, ::-1]
        center_line_pts = center_line_pts.reshape((-1, 1, 2))

        cv2.polylines(img, [center_line_pts], isClosed=True, color=0, thickness=expand_width, lineType=cv2.LINE_AA)
        img = img & self.canvas
        return img
    
    @classmethod
    def drawLineAndPoints(cls, img, pts:np.ndarray, isImageCoord:bool=False):

        pts = pts[:, ::-1].astype(np.int32)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        cv2.line(img, pts[0], pts[-1], (255, 255, 0), 1, cv2.LINE_AA)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i+1], (255, 255, 0), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
        
        
        return img
    
    def generate_constrained_points(self, n: int, min_dist: float, max_range: float = 200) -> np.ndarray:
        
        points = []
        max_attempts = n * 100
        
        for _ in range(max_attempts):
            if len(points) >= n:
                break

            candidate = self.rng.random(2) * max_range - (max_range / 2)
            
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
            print(f"Only {len(current_skeleton)}/{num_pts} points met the distance constraint.")

        return np.array(current_skeleton)


