https://huggingface.co/bhagyashrideshmukh72/yieldprediction/blob/main/NEW_trainedCropHeightModel_Customized.onnx
import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sugarcane Yield Prediction", layout="centered")

# -------------------------
# Download and cache ONNX model
# -------------------------
@st.cache_data
def download_model(url, output="model.onnx"):
    if not os.path.exists(output):
        st.info("Downloading ONNX model...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded!")
    return output

# -------------------------
# Load ONNX model
# -------------------------
@st.cache_data
def load_model(path):
    return ort.InferenceSession(path)

# Replace with your Hugging Face direct download link
model_url = "https://huggingface.co/bhagyashrideshmukh72/yieldprediction/blob/main/NEW_trainedCropHeightModel_Customized.onnx"
model_path = download_model(model_url)
session = load_model(model_path)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Sugarcane Yield Prediction App")
st.write("Upload an image of your sugarcane farm to predict yield.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------
    # Preprocess image
    # -------------------------
    # Resize to 224x224 (adjust if your model uses a different size)
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)

    # Normalize if required (example: divide by 255)
    image_array = image_array / 255.0

    # Add batch dimension
    input_data = np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)

    # -------------------------
    # Make prediction
    # -------------------------
    try:
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})
        st.success(f"Predicted Yield: {result[0][0]:.2f} kg/ha")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
