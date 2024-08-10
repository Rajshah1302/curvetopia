import cv2
import numpy as np

def is_star(contour):
    """
    Determine if the contour represents a star shape based on its properties.
    """
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices > 5:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if circularity < 0.5:
            return True

    return False

def is_circle(contour):
    """
    Determine if the contour represents a circle shape based on its properties.
    """
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Adjust the circularity threshold as needed
    return circularity > 0.8
