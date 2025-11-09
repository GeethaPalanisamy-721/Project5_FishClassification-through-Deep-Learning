import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# ============================================
# PATH CONFIGURATION
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# Fine-tuned models only
MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0_finetuned_v4_final.keras",
    "MobileNetV2": "mobilenetv2_finetuned.keras",
    "InceptionV3": "inceptionv3_finetuned.keras",
    "ResNet50": "resnet50_finetuned.keras",
    "VGG16": "vgg16_finetuned.keras"
}

# ============================================
# 🧠 LOAD CLASS NAMES
# ============================================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    CLASS_NAMES = list(class_indices.keys())
else:
    st.warning("⚠️ Could not load class_indices.json — using default placeholders.")
    CLASS_NAMES = [f"Class {i}" for i in range(11)]  # fallback if missing

# ============================================
# ⚡ LOAD MODELS (cache for speed)
# ============================================
@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with st.spinner(f"Loading {name}..."):
                models[name] = tf.keras.models.load_model(path)
        else:
            st.warning(f"⚠️ {filename} not found — skipping.")
    return models

MODELS = load_models()

# ============================================
# 🖼️ PREPROCESS IMAGE
# ============================================
def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img, img_array

# ============================================
# 🔍 PREDICT USING MODEL
# ============================================
def predict_image(model, img_array):
    preds = model.predict(img_array)
    top5_idx = preds[0].argsort()[-5:][::-1]
    top5_probs = preds[0][top5_idx]
    top5_labels = [CLASS_NAMES[i] for i in top5_idx]
    top_pred = top5_labels[0]
    top_conf = top5_probs[0] * 100
    return top_pred, top_conf, top5_labels, top5_probs

# ============================================
# 🎨 STREAMLIT UI
# ============================================
st.set_page_config(page_title="Fish Classification App", layout="wide")

st.title("🐟 Fish Species Classification Dashboard")

col1, col2 = st.columns([0.45, 0.55], gap="large")

# ---------- LEFT COLUMN ----------
with col1:
    st.subheader("⚙️ Model Selection & Prediction")

    model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    selected_model = MODELS.get(model_name)

    uploaded_file = st.file_uploader("Upload a Fish Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and selected_model is not None:
        img, img_array = preprocess_image(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        st.markdown("### 🔎 Prediction Results")
        pred_label, pred_conf, top5_labels, top5_probs = predict_image(selected_model, img_array)

        st.success(f"**Predicted Fish:** {pred_label}")
        st.info(f"**Confidence:** {pred_conf:.2f}%")

# ---------- RIGHT COLUMN ----------
with col2:
    st.subheader("📊 Model Insights & Comparison")

    if uploaded_file is not None:
        # ---- Top-5 Predictions Bar Chart ----
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top5_labels[::-1], top5_probs[::-1] * 100, color='skyblue')
        ax.set_xlabel("Confidence (%)")
        ax.set_title(f"Top-5 Predictions - {model_name}")
        st.pyplot(fig)

        # ---- Compare Across All Models ----
        st.markdown("### 🤖 Confidence Comparison Across Models")

        results = []
        for m_name, m_model in MODELS.items():
            _, conf, _, _ = predict_image(m_model, img_array)
            results.append({"Model": m_name, "Confidence (%)": round(conf, 2)})

        df = pd.DataFrame(results).sort_values("Confidence (%)", ascending=False)
        st.table(df.set_index("Model"))

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Developed for Fish Image Classification Project 🐠 | Streamlit App")

