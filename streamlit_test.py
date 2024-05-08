import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras import backend as K

NB_IMAGES = 20
selected_id = 1
img_paths = 'static/data/img/'
mask_paths = 'static/data/mask/'

cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
 }

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

# Fonctions loss
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.cast(K.flatten(y_true), K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

# Prépare les données pour la segmentation avec model.predict()
def get_data_prepared(path_X, dim):
    X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), dim)])
    X = X / 255

    return X

# Prépare l'image pour la segmentation
def prepare_img(img, dim):
    X = np.array([cv2.resize(np.array(img), dim)])
    X = X / 255

    return X

# Recupère les chemins d'accès des fichiers
def getPathFiles():
    path_files = []

    # img set
    for file in os.listdir(img_paths):
        path_files.append(file.replace('leftImg8bit.png',''))

    return path_files
def predict_image(selected_id):
    img_path = img_paths + path_files[selected_id-1] + 'leftImg8bit.png'

    img = get_data_prepared(img_path, (256,256))
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

    cv2.imwrite('static/data/predict/mask_predicted.png', cv2.resize(m, (400,200)))

    background = cv2.imread('static/data/predict/img.png')
    overlay = cv2.imread('static/data/predict/mask_predicted.png')

    added_image = cv2.addWeighted(background,1,overlay,0.6,0)

    cv2.imwrite('static/data/predict/combined.png', added_image)

    return 'static/data/predict/combined.png'


model = tf.lite.Interpreter(model_path='model/ResNet50_U-Net_basic.tflite')
model.allocate_tensors()

path_files = getPathFiles()

def process_image(selected_id):
    img_path = img_paths + path_files[selected_id-1] + 'leftImg8bit.png'
    mask_path = mask_paths + path_files[selected_id-1] + 'gtFine_labelIds.png'

    img = cv2.resize(cv2.imread(img_path), (400, 200))
    mask = cv2.resize(cv2.imread(mask_path), (400, 200))
    mask = np.squeeze(mask[:,:,0])
    mask_labelids = np.zeros((mask.shape[0], mask.shape[1], len(cats_id)))

    for i in range(-1, 34):
        for cat in cats:
            if i in cats[cat]:
                mask_labelids[:,:,cats_id[cat]] = np.logical_or(mask_labelids[:,:,cats_id[cat]],(mask==i))
                break

    mask_labelids = np.array(np.argmax(mask_labelids,axis=2), dtype='uint8')

    m = np.empty((mask_labelids.shape[0], mask_labelids.shape[1], 3), dtype='uint8')
    for i in range(mask_labelids.shape[0]):
        for j in range(mask_labelids.shape[1]):
            m[i][j] = cats_colors[mask_labelids[i][j]]

    cv2.imwrite('static/data/predict/img.png', img)
    cv2.imwrite('static/data/predict/mask.png', m)

    return 'static/data/predict/img.png', 'static/data/predict/mask.png'

st.title('Image Prediction')

selected_id = st.slider('Select Image ID', 1, NB_IMAGES, 1)

if st.button('Process'):
    img_path, mask_path = process_image(selected_id)
    st.image(img_path, caption='Original Image')
    st.image(mask_path, caption='Mask Image')

if st.button('Predict'):
    result = predict_image(selected_id)
    st.image(result, caption='Predicted Image')