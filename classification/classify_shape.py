import cv2
import numpy as np
from .is_shapes import is_star, is_circle
from .classify_triangle import classify_triangle

def is_straight_line(contour):
    """
    Determine if the contour represents a straight line.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If the number of vertices is 2, it's a straight line
    return len(approx) == 2

def is_ellipse(contour):
    """
    Determine if the contour represents an ellipse.
    """
    # Fit an ellipse to the contour points
    contour_points = contour.reshape(-1, 2)
    if len(contour_points) >= 5:
        ellipse = cv2.fitEllipse(contour_points)
        return True
    return False

def classify_shape(contour):
    """
    Classify the shape based on contour approximation.
    """
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if is_straight_line(contour):
        return "Straight Line"
    elif num_vertices == 3:
        return classify_triangle(contour)
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return "Square" if 0.95 < aspect_ratio < 1.05 else "Rectangle"
    elif num_vertices > 6:
        if is_star(contour):
            return "Star"
        return "Polygon"
    return "Unknown Shape"
