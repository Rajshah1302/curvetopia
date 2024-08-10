import numpy as np
from scipy.interpolate import UnivariateSpline
from classification.is_shapes import is_star

def smooth_contour(contour):
    """
    Smooth the contour points using spline interpolation.
    """
    contour_points = contour.reshape(-1, 2)
    if len(contour_points) < 6 or is_star(contour):
        return contour  # Not enough points to smooth

    # Create a parameter t which ranges from 0 to 1
    t = np.linspace(0, 1, len(contour_points))
    
    # Spline interpolation for x and y coordinates
    spl_x = UnivariateSpline(t, contour_points[:, 0], s=0, k=3)
    spl_y = UnivariateSpline(t, contour_points[:, 1], s=0, k=3)
    
    # Generate new parameter t values and corresponding smoothed points
    t_new = np.linspace(0, 1, 100)  # Number of points in smoothed contour
    smooth_x = spl_x(t_new)
    smooth_y = spl_y(t_new)
    
    smooth_contour_points = np.column_stack((smooth_x, smooth_y))
    return smooth_contour_points.astype(np.int32).reshape(-1, 1, 2)