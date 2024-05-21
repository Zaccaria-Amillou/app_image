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
    response = requests.post(f'https://zacam.pythonanywhere.com/prediction/', data={'file': id})

    # Check if the request was successful
    if response.status_code == 200:
        st.success('Image processed successfully!')

        # Create two columns for the first row
        col1, col2 = st.columns(2)

        # Display the real image and mask
        response = requests.get(f'https://zacam.pythonanywhere.com/static/data/predict/img.png')
        image = Image.open(io.BytesIO(response.content))
        col1.image(image, caption='Real Image', use_column_width=True)

        response = requests.get(f'https://zacam.pythonanywhere.com/static/data/predict/mask.png')
        image = Image.open(io.BytesIO(response.content))
        col2.image(image, caption='Real Mask', use_column_width=True)

        # Make a POST request to the Flask app to get the prediction
        response = requests.post(f'https://zacam.pythonanywhere.com/predict/')

        # Check if the request was successful
        if response.status_code == 200:
            st.success('Prediction submitted successfully!')

            # Create two columns for the second row
            col3, col4 = st.columns(2)

            # Display the predicted mask and combined image
            response = requests.get(f'https://zacam.pythonanywhere.com/static/data/predict/mask_predicted.png')
            image = Image.open(io.BytesIO(response.content))
            col3.image(image, caption='Predicted Mask', use_column_width=True)

            response = requests.get(f'https://zacam.pythonanywhere.com/static/data/predict/combined.png')
            image = Image.open(io.BytesIO(response.content))
            col4.image(image, caption='Combined Image', use_column_width=True)
        else:
            st.error('Failed to submit prediction.')
    else:
        st.error('Failed to process image.')