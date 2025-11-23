

import matplotlib.pyplot as plt
import numpy as np
from random import Random

class Generator():
    def __init__(self, seed = None):
        self.mapSize = None
        self.borderWidth = None
        if not isinstance(seed, (None, int)):
            seed = 1234
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)
        
    def createFramework(self, mapSize, borderWidth = 0):
        pass


def generate_constrained_points(n: int, min_dist: float, max_range: float = 200) -> np.ndarray:
    
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

def connect_via_angular_sort(points: np.ndarray) -> np.ndarray:

    if len(points) < 3:
        return points

    center = np.mean(points, axis=0)
    
    vectors = points - center
    
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) 
    
    sorted_indices = np.argsort(angles)
    
    sorted_path_skeleton = points[sorted_indices]
    
    return sorted_path_skeleton

def get_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
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

def optimize_path_skeleton(skeleton: np.ndarray, min_angle_degrees: float) -> np.ndarray:
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
            
            angle = get_angle(p_prev, p_curr, p_next)
            
            if angle < min_angle_radians:
                current_skeleton.pop(i)
                n -= 1 
                optimized = False
            else:
                i += 1
                
        if not optimized:
            i = 0
    if len(current_skeleton) < num_pts:
        print(f"Warning: Only {len(current_skeleton)}/{num_pts} points met the distance constraint.")

    return np.array(current_skeleton)

N_SKELETON_POINTS = 130
MIN_DISTANCE = 500
MAX_RANGE = 2000
MIN_ANGLE_DEGREES = 40

try:
    while(True):
        skeleton_points = generate_constrained_points(N_SKELETON_POINTS, MIN_DISTANCE, MAX_RANGE)
        ordered_skeleton = connect_via_angular_sort(skeleton_points)
        optimized_skeleton = optimize_path_skeleton(ordered_skeleton, MIN_ANGLE_DEGREES)

        if(len(optimized_skeleton) >= 3):
            break

    fig, ax = plt.subplots(figsize=(10, 10))

    closed_skeleton_for_plot = np.vstack([optimized_skeleton, optimized_skeleton[0]])
    ax.plot(closed_skeleton_for_plot[:, 0], closed_skeleton_for_plot[:, 1], 'g--', linewidth=1, alpha=0.5, label='Skeleton (Angle Sorted)')

    ax.plot(optimized_skeleton[:, 0], optimized_skeleton[:, 1], 'bo', markersize=6, label='Skeleton Points')
    
    ax.set_title(f"trace track")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis('equal')
    plt.legend()
    plt.show()

except ValueError as e:
    print(f"Error: {e}")