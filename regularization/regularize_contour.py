import cv2
import numpy as np
from classification.is_shapes import is_circle, is_star
from classification.classify_shape import classify_shape
from regularization.regularize_circle import regularize_circle
from regularization.smooth_contour import smooth_contour
def is_straight_line(contour):
    """
    Determine if the contour represents a straight line based on its properties.
    """
    contour_points = contour.reshape(-1, 2)
    if len(contour_points) < 2:
        return False

    # Fit a line to the points using least squares method
    x_coords = contour_points[:, 0]
    y_coords = contour_points[:, 1]
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    # Calculate the distance of all points to the fitted line
    distances = np.abs(y_coords - (m * x_coords + c)) / np.sqrt(m**2 + 1)
    
    # Determine if all points are approximately on the line
    return np.all(distances < 1)  # Adjust the threshold as needed

def is_complex_shape(contour):
    """
    Determine if the contour represents a complex shape that should not be regularized.
    """
    contour_points = contour.reshape(-1, 2)
    num_vertices = len(contour_points)
    area = cv2.contourArea(contour)
    
    if is_straight_line(contour):
        return True  # Treat straight lines as complex shapes to avoid regularization
    
    if num_vertices > 10 or area < 100:  # Adjust these thresholds as needed
        return True
    
    return False

def regularize_contour(contour):
    """
    Regularize the contour using Douglas-Peucker algorithm for non-circles
    and midpoint algorithm for circles, with added path smoothing.
    """
    shape_type = classify_shape(contour)
    if is_circle(contour) or shape_type=='Polygon':
        return regularize_circle(contour)
    elif is_star(contour) or shape_type == 'Rectangle':
        # Regularize stars by smoothing
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return smooth_contour(approx)
    else:
        # Avoid regularizing complex shapes and straight lines
        if is_complex_shape(contour):
            return contour  # Return the original contour without regularization
        
        # Apply smoothing to simpler shapes
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return smooth_contour(approx)
