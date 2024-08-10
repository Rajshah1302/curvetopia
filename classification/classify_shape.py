import cv2
from .is_shapes import is_star, is_circle
from .classify_triangle import classify_triangle

def classify_shape(contour):
    """
    Classify the shape based on contour approximation.
    """
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return classify_triangle(contour)
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return "Square" if 0.95 < aspect_ratio < 1.05 else "Rectangle"
    elif num_vertices == 5:
        return "Pentagon"
    elif num_vertices == 6:
        return "Hexagon"
    elif num_vertices > 6:
        if is_star(contour):
            return "Star"
        return "Polygon"
    return "Unknown Shape"
