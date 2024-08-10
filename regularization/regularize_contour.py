import cv2
from classification.is_shapes import is_circle
from regularization.regularize_circle import regularize_circle
from regularization.smooth_contour import smooth_contour

def regularize_contour(contour):
    """
    Regularize the contour using Douglas-Peucker algorithm for non-circles
    and midpoint algorithm for circles, with added path smoothing.
    """
    if is_circle(contour):
        return regularize_circle(contour)
    else:
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return smooth_contour(approx)
