import json
import os

import gdown
import numpy as np
import streamlit as st
from PIL import Image
from keras.src.saving import load_model


# Function to download the file from Google Drive
def download_file_from_url(url, destination):
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    gdown.download(url, destination, quiet=False)

# URL of the model file in Google Drive
model_url = 'https://drive.google.com/uc?export=download&id=1xBJlsr8nEY7vqKI8SQFu5vk73c9XoTut'  # Model file ID
class_indices_url = 'https://drive.google.com/uc?export=download&id=1fI4tX3iBfQkr80QKzQng8xOvOqJqVrpU'  # Class indices file ID

# Download the model and class indices if not already downloaded
model_path = 'model/model.h5'
class_indices_path = 'class_indices.json'

if not os.path.exists(model_path):
    download_file_from_url(model_url, model_path)

if not os.path.exists(class_indices_path):
    download_file_from_url(class_indices_url, class_indices_path)

# Load the model
model = load_model(model_path)

# Load the class indices
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Streamlit App
st.title("Plant Disease Prediction")
st.write("Upload an image of the plant leaf to predict the disease")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    img = img.resize((224, 224))  # Resize the image to match the model's input shape
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the class label
    predicted_class_label = class_indices[str(predicted_class_index)]

    # Show the prediction result
    st.write(f"Predicted class: {predicted_class_label}")


    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the class label
    predicted_class_label = class_indices[str(predicted_class_index)]

    # Show the prediction result
    st.write(f"Predicted class: {predicted_class_label}")
