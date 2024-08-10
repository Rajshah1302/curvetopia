# Curvetopia

## Overview

This project, **Curvetopia**, provides a Python-based solution for detecting and regularizing contours from CSV files containing XY coordinates. It includes functionality for regularizing contours, classifying shapes, and plotting the results. A Flask-based web frontend allows users to upload CSV files, which are processed and displayed as plots in the browser.

## Features

- **Shape Classification**: Identifies and classifies various shapes including triangles, rectangles, pentagons, hexagons, circles, and stars.
- **Contour Regularization**: Smoothens and regularizes contours using various algorithms.
- **CSV File Handling**: Reads and parses CSV files containing XY coordinates of shapes.
- **Web Interface**: Upload CSV files through a web interface and view the processed plots.

## Project Structure

```
curve_analysis/
├── __init__.py
├── classification/
│   ├── __init__.py
│   ├── classify_shape.py
│   ├── classify_triangle.py
│   └── is_shapes.py
├── regularization/
│   ├── __init__.py
│   ├── regularize_contour.py
│   ├── regularize_circle.py
│   └── smooth_contour.py
├── io/
│   ├── __init__.py
│   └── read_csv.py
├── utils/
│   ├── __init__.py
│   └── calculate_triangle_angles.py
├── static/
│   └── styles.css
├── templates/
│   ├── index.html
│   └── plot.html
├── app.py
└── main.py
```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/curve_analysis.git
   cd curve_analysis
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content:

   ```plaintext
   Flask
   numpy
   matplotlib
   opencv-python
   scipy
   ```

4. **Setup**

   Make sure your project directory structure matches the above layout. Ensure that the `uploads` directory exists to store uploaded files:

   ```bash
   mkdir uploads
   ```

## Usage

1. **Run the Flask Application**

   ```bash
   python app.py
   ```

   The application will start and be accessible at `http://127.0.0.1:5000/`.

2. **Upload a CSV File**

   - Open your web browser and navigate to `http://127.0.0.1:5000/`.
   - Use the form to upload a CSV file containing XY coordinates of shapes.
   - The processed shapes will be displayed on a new page.

## Functions Overview

- **`read_csv(csv_path)`**: Reads and parses a CSV file into paths of XY coordinates.
- **`classify_shape(contour)`**: Classifies shapes based on contour properties.
- **`regularize_contour(contour)`**: Regularizes contours using smoothing algorithms.
- **`plot(paths_XYs)`**: Generates and displays plots of the shapes.
---
