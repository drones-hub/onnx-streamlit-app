import streamlit as st
import os
import onnxruntime as ort
import numpy as np
import requests

st.set_page_config(page_title="ONNX Model Prediction", layout="centered")

# -------------------------
# Function to download model once
# -------------------------
@st.cache_data(show_spinner=True)
def download_model(url, output="model.onnx"):
    if not os.path.exists(output):
        st.info("Downloading ONNX model...")
        r = requests.get(url, stream=True)
        r.raise_for_status()  # ensure we get an error if download fails
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded!")
    return output

# -------------------------
# Use a direct download link (Hugging Face, GitHub LFS, or Dropbox dl=1)
# -------------------------
model_url = "https://www.dropbox.com/s/2gngsmey4f56d2bxk4gjb/NEW_trainedCropHeightModel_Customized.onnx?dl=1"
model_path = download_model(model_url)

# -------------------------
# Load ONNX model
# -------------------------
@st.cache_data(show_spinner=True)
def load_model(path):
    return ort.InferenceSession(path)

session = load_model(model_path)

# -------------------------
# Streamlit UI
# -------------------------
st.title("ONNX Model Prediction App")

crop_age = st.selectbox("Crop Age (months)", [8, 10])
visual_category = st.selectbox("Visual Category", ["Sugarcane", "Non-sugarcane", "Mixed"])
yield_class = st.selectbox("Yield Class", ["Poor", "Good", "Very Good"])

visual_map = {"Sugarcane": 0, "Non-sugarcane": 1, "Mixed": 2}
yield_map = {"Poor": 0, "Good": 1, "Very Good": 2}

input_data = np.array([[crop_age, visual_map[visual_category], yield_map[yield_class]]], dtype=np.float32)

if st.button("Predict"):
    input_name = session.get_inputs()[0].name
    try:
        result = session.run(None, {input_name: input_data})
        st.success(f"Prediction: {result[0][0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
