import numpy as np
from utils.calculate_triangle_angles import calculate_triangle_angles

def classify_triangle(contour):
    """
    Classify the triangle based on its properties.
    """
    contour_points = contour.reshape(-1, 2)
    
    if contour_points.shape[0] != 3:
        print("Contour does not have 3 vertices: ", contour_points.shape[0])
        return "Not a Triangle"
  
    # Calculate side lengths
    side_lengths = np.sqrt(np.sum(np.diff(contour_points, axis=0, append=[contour_points[0]])**2, axis=1))
    
    # Ensure we have exactly 3 side lengths
    if len(side_lengths) != 3:
        print("Side lengths array has incorrect length: ", len(side_lengths))
        return "Not a Triangle"
    
    angles = calculate_triangle_angles(side_lengths)
    
    # Check if it's an equilateral, isosceles, or scalene triangle
    if np.allclose(side_lengths[0], side_lengths[1]) and np.allclose(side_lengths[1], side_lengths[2]):
        return "Equilateral Triangle"
    elif np.allclose(side_lengths[0], side_lengths[1]) or np.allclose(side_lengths[1], side_lengths[2]) or np.allclose(side_lengths[2], side_lengths[0]):
        return "Isosceles Triangle"
    else:
        return "Scalene Triangle"
