import numpy as np

def calculate_triangle_angles(side_lengths):
    """
    Calculate angles of a triangle given its side lengths using the law of cosines.
    """
    a, b, c = side_lengths
    angle_A = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
    angle_B = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
    angle_C = np.pi - angle_A - angle_B
    return np.degrees([angle_A, angle_B, angle_C])
