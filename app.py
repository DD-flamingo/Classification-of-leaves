import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os


# Page configuration
st.set_page_config(
    page_title="Leaf Species Identification",
    page_icon="🌿",
    layout="centered"
)


# Load trained model
MODEL_PATH = os.path.join("models", "tree_species_model.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# Class names (MUST match training order)
class_names = ["Mango", "Sandalwood"]


# UI Header
st.title("🌿 Tree / Leaf Species Identification")
st.markdown(
    "Upload a **leaf image** and the model will predict the **tree species**."
)

st.divider()


# Image uploader
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# Prediction pipeline
if uploaded_file is not None:
    # Load image
    image_pil = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image_pil,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess
    img = image_pil.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.divider()

    # Results
    st.success(f"✅ **Predicted Species:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")


# Footer
st.caption("Developed for academic & research purposes")
