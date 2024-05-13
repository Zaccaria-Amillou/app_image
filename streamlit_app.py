# streamlit_app.py
import streamlit as st
import requests
import io
import os
from PIL import Image

st.title('Image Segmentation')

id = st.slider('Choose an image id', min_value=1, max_value=21)  

if st.button('Submit'):
    # Make a POST request to the Flask app to process the image
    response = requests.post(f'http://127.0.0.1:5000/prediction/', data={'file': id})

    # Check if the request was successful
    if response.status_code == 200:
        st.success('Image processed successfully!')

        # Display the real image and mask
        response = requests.get(f'http://127.0.0.1:5000/static/data/predict/img.png')
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption='Real Image', use_column_width=True)

        response = requests.get(f'http://127.0.0.1:5000/static/data/predict/mask.png')
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption='Real Mask', use_column_width=True)

        # Make a POST request to the Flask app to get the prediction
        response = requests.post(f'http://127.0.0.1:5000/predict/')

        # Check if the request was successful
        if response.status_code == 200:
            st.success('Prediction submitted successfully!')

            # Display the predicted mask and combined image
            response = requests.get(f'http://127.0.0.1:5000/static/data/predict/mask_predicted.png')
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption='Predicted Mask', use_column_width=True)

            response = requests.get(f'http://127.0.0.1:5000/static/data/predict/combined.png')
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption='Combined Image', use_column_width=True)
        else:
            st.error('Failed to submit prediction.')
    else:
        st.error('Failed to process image.')