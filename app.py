import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection (VGG16)", layout="centered")
st.title("Plant Disease Classification (VGG16 - Transfer Learning)")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vgg16_plant_disease_model.h5")

model = load_model()

class_names = [
    'Tomato Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot'
]

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Disease: {predicted_class}")
