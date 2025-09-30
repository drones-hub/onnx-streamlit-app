import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np

# Download model once and cache
@st.cache_data
def download_model(url, output="model.onnx"):
    if not os.path.exists(output):
        r = requests.get(url)
        with open(output, "wb") as f:
            f.write(r.content)
    return output

dropbox_url = "https://www.dropbox.com/s/2gngsmey4f56d2bxk4gjb/NEW_trainedCropHeightModel_Customized.onnx?dl=1"
model_path = download_model(dropbox_url)

# Load ONNX model
@st.cache_data
def load_model(path):
    return ort.InferenceSession(path)

session = load_model(model_path)

# Streamlit UI
st.title("ONNX Model Prediction App")

crop_age = st.selectbox("Crop Age (months)", [8, 10])
visual_category = st.selectbox("Visual Category", ["Sugarcane", "Non-sugarcane", "Mixed"])
yield_class = st.selectbox("Yield Class", ["Poor", "Good", "Very Good"])

visual_map = {"Sugarcane": 0, "Non-sugarcane": 1, "Mixed": 2}
yield_map = {"Poor": 0, "Good": 1, "Very Good": 2}

input_data = np.array([[crop_age, visual_map[visual_category], yield_map[yield_class]]], dtype=np.float32)

if st.button("Predict"):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})
    st.success(f"Prediction: {result[0][0]}")
