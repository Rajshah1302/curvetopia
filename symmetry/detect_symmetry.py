import numpy as np

def detect_symmetry_lines(contour):
    """
    Detect vertical and horizontal symmetry lines for a given contour.
    """
    contour_points = contour.reshape(-1, 2)
    
    # Bounding box of the contour
    x_min, y_min = np.min(contour_points, axis=0)
    x_max, y_max = np.max(contour_points, axis=0)
    
    # Vertical and horizontal lines of symmetry
    vertical_line_x = (x_min + x_max) / 2
    horizontal_line_y = (y_min + y_max) / 2
    
    return vertical_line_x, horizontal_line_y
