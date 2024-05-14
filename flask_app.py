import os
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from flask import Flask, request, send_file, render_template, redirect, url_for
from keras.models import load_model
from keras.losses import categorical_crossentropy
import subprocess
from keras import backend as K

# Nombre des images
NB_IMAGES = 20

# definition des variables
selected_id = 1
img_paths = 'static/data/img/'
mask_paths = 'static/data/mask/'

# Catégories des images
cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
 }

# Idéntifiant des catégories
cats_id = {
 'void': (0),
 'flat': (1),
 'construction': (2),
 'object': (3),
 'nature': (4),
 'sky': (5),
 'human':(6),
 'vehicle': (7)
}

# Couleurs des catégories
cats_colors = {
 0: (0,0,0),
 1: (50,50,50),
 2: (150,150,150),
 3: (255,0,0),
 4: (0,255,0),
 5: (0,0,255),
 6:(200,200,0),
 7: (150,0,200)
}

def get_data_prepared(path_X, dim):
    """Prépare les données pour la segmentation."""
    X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), dim)])
    X = X / 255

    return X

def prepare_img(img, dim):
    """Prépare l'image pour la segmentation."""
    X = np.array([cv2.resize(np.array(img), dim)])
    X = X / 255

    return X

# Recupère les chemins d'accès des fichiers
def getPathFiles():
    """Récupère les chemins d'accès des fichiers."""
    path_files = []

    # img set
    for file in os.listdir(img_paths):
        path_files.append(file.replace('leftImg8bit.png',''))

    return path_files


path_files = getPathFiles()

app = Flask(__name__)

# chargement du modèle
model = tf.lite.Interpreter(model_path='model/ResNet50_U-Net_basic.tflite')
model.allocate_tensors()

@app.route('/', methods=['GET','POST'])
def homepage():
    """Route pour la page d'accueil."""
    return render_template('index.html')

from flask import redirect

@app.route('/prediction', methods=['GET', 'POST'])
@app.route('/prediction/', methods=['GET', 'POST'])
def redirectImage():
    """Route for redirecting an image."""
    return redirect("http://localhost:8501", code=302)
    

@app.route('/segment', methods=['GET', 'POST'])
@app.route('/segment/', methods=['GET', 'POST'])
def segmentImage():
    """Route pour segmenter une image."""
    if request.method == 'POST':
        # Handle the POST request 
        file = request.files.get('image')

        if not file:
            return "No image uploaded", 400

        # Open the image file
        image = Image.open(file.stream)

        # Resize the image
        image = image.resize((256, 256))

        # Save the resized original image to a static directory
        image.save('static/data/original/original.png')

        img = prepare_img(image, (256,256))
        img = img.astype('float32')  # Convert the input tensor to float32
        input_details = model.get_input_details()
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        y_pred = model.get_tensor(model.get_output_details()[0]['index'])
        y_pred_argmax=np.argmax(y_pred, axis=3)

        m = np.empty((y_pred_argmax[0].shape[0],y_pred_argmax[0].shape[1],3), dtype='uint8')
        for i in range(y_pred_argmax[0].shape[0]):
            for j in range(y_pred_argmax[0].shape[1]):
                m[i][j] = cats_colors[y_pred_argmax[0][i][j]]

        m = cv2.resize(m, (256,256))

        im_bgr = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(im_bgr)

        # Save the segmented image to a static directory
        img.save('static/data/segmented/segmented.png')

        # Identify the categories present in the image
        unique_categories = np.unique(y_pred_argmax)
        present_categories = [key for key, value in cats_id.items() if value in unique_categories]

        # Redirect to the results page
        return redirect(url_for('results', categories=present_categories))
    else:
        # Handle the GET request here
        return render_template('segment.html')

@app.route('/results', methods=['GET'])
def results():
    """Route pour les résultats."""
    categories = request.args.getlist('categories')
    return render_template('results.html', categories=categories)

if __name__ == "__main__":
    app.run()