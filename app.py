import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Garbage Classifier", layout="centered")

MODEL_URL = "https://huggingface.co/2005-wajahat/gaarbage-classifier-v2/resolve/main/garabage-classifier-v3.keras"

class_names = [
    'Unknown',
    'glass',
    'metal',
    'organic_waste',
    'paper_cardboard',
    'plastic',
    'textiles',
    'trash'
]

IMG_SIZE = (300, 300)
CONFIDENCE_THRESHOLD = 0.6

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file(
        "model.keras",
        MODEL_URL
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# UI
# =========================
st.title("♻️ Garbage Classifier AI")
st.write("Upload an image or use camera to classify waste.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Take a picture")

image = None

if camera_image is not None:
    image = Image.open(camera_image)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# =========================
# PREDICTION
# =========================
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img = preprocess_image(image)

    with st.spinner("Analyzing..."):
        pred = model.predict(img)

    pred = pred[0]
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    # =========================
    # SMART OUTPUT
    # =========================
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Not sure what this is (Low confidence)")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"Prediction: **{class_names[class_index]}**")
        st.info(f"Confidence: {confidence*100:.2f}%")

    # =========================
    # CHART
    # =========================
    chart_data = pd.DataFrame({
        "Class": class_names,
        "Probability": pred
    }).set_index("Class")

    st.bar_chart(chart_data)
