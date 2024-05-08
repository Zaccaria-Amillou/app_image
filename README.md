# Image Segmentation Flask Application

This is a Flask application for image segmentation using a pre-trained model.

## Project Structure

The project is structured as follows:

- `flask_app.py`: This is the main file that contains the Flask application.
- `model/ResNet50_U-Net_basic.tflite`: This is the pre-trained model used for image segmentation.
- `static/data`: This directory contains the images and masks used by the application.

## Features

The application provides the following features:

- Image Segmentation: The application can segment images into different categories.
- Image Prediction: The application can predict the segmentation of an image.
- Image Results: The application can display the results of the segmentation.

## Installation

To install the application, follow these steps:

1. Clone the repository.
2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python flask_app.py
```
## Usage
To use the application, navigate to http://localhost:5000 in your web browser.
