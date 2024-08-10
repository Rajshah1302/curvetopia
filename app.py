from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
from csv_io.read_csv import read_csv
from classification.classify_shape import classify_shape
from regularization.regularize_contour import regularize_contour
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Configure upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        paths_XYs = read_csv(filename)
        img = plot_and_save(paths_XYs)
        return render_template('plot.html', img_data=img)
    return redirect(request.url)

def plot_and_save(paths_XYs):
    """
    Plot the shapes and save the plot to an image in memory.
    """
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))

    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define colors
    
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]  # Choose color based on index
        for XY in XYs:
            contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
            regularized_contour = regularize_contour(contour)
            shape_type = classify_shape(regularized_contour)
            
            # Directly plot the contour using matplotlib
            contour_points = regularized_contour.reshape(-1, 2)
            # Ensure the contour is closed
            if not np.array_equal(contour_points[0], contour_points[-1]):
                contour_points = np.vstack([contour_points, contour_points[0]])
                
            ax.plot(contour_points[:, 0], contour_points[:, 1], c=c, linewidth=2)
            
           
    
    ax.set_aspect('equal')  # Ensure aspect ratio is equal
    ax.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return img_data

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
