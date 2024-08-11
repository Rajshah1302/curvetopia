import matplotlib.pyplot as plt
import numpy as np
from classification.classify_shape import classify_shape
from regularization.regularize_contour import regularize_contour

# Define some example colors if not already defined
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add or modify as needed

def read_csv(csv_path):
    """
    Read CSV file and parse paths with XY coordinates.
    """
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []

        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

def plot(paths_XYs):
    """
    Plot the shapes from the paths_XYs data.
    """
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))

    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]  # Choose color based on index
        for XY in XYs:
            contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
            regularized_contour = regularize_contour(contour)
            shape_type = classify_shape(regularized_contour)

            if shape_type == "Straight line":
                # Directly plot the line using the start and end points
                x0, y0 = XY[0]
                x1, y1 = XY[-1]
                ax.plot([x0, x1], [y0, y1], c=c, linewidth=2, label=shape_type)
            elif shape_type == 'Unknown Shape' or shape_type == 'Not a Triangle':
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, label=shape_type)
            else:
                # Plot the regularized contour for other shapes
                # regularized_contour = regularize_contour(contour)
                contour_points = regularized_contour.reshape(-1, 2)
                if shape_type == "Straight Line":
                    x0, y0 = XY[0]
                    x1, y1 = XY[-1]
                    ax.plot([x0, x1], [y0, y1], c=c, linewidth=2, label=shape_type)
                # Ensure the contour is closed
                else:
                    if not np.array_equal(contour_points[0], contour_points[-1]):
                        contour_points = np.vstack([contour_points, contour_points[0]])
                    ax.plot(contour_points[:, 0], contour_points[:, 1], c=c, linewidth=2, label=shape_type)


            # Annotate the shape type
            centroid = np.mean(XY, axis=0)
            ax.text(centroid[0], centroid[1], shape_type, fontsize=12, ha='center')

    ax.set_aspect('equal')  # Ensure aspect ratio is equal
    ax.legend()
    plt.show()  # Display the plot

# Example usage
if __name__ == "__main__":
    csv_path = r'D:\python_projects\frag0.csv'  
    paths_XYs = read_csv(csv_path)

    # Plot shapes
    plot(paths_XYs)
