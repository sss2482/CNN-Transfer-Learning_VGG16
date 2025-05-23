import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import gdown

@st.cache_resource
def load_trained_model():
    model_path = "vgg16_plant_disease_model.h5"
    file_id = "1Kns43o5nbU6RruFtm3TEn-CShKY4QWqT"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)
    return model

model = load_trained_model()

label_dict = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Tomato_Early_blight',
    5: 'Tomato_Leaf_Mold'
}

st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to detect plant disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"### 🧪 Prediction: `{label_dict[predicted_class]}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
