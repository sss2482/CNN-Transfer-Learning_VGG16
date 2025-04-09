import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("vgg16_plant_disease_model.h5")

# Class labels
class_names = [
    'Tomato Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot'
]

# Streamlit app UI
st.title(" Plant Disease Classification (Transfer Learning - VGG16)")
st.write("Upload a tomato leaf image to detect the disease.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 150, 150, 3))

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Prediction: {predicted_class}")
