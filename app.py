import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
from PIL import Image

# -----------------------------
# 1. Download ONNX model
# -----------------------------
model_url = "https://huggingface.co/bhagyashrideshmukh72/yieldprediction/blob/main/NEW_trainedCropHeightModel_Customized.onnx"

@st.cache_data
def download_model(url, output="model.onnx"):
    if not os.path.exists(output):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return output

model_path = download_model(model_url)
session = ort.InferenceSession(model_path)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Crop Height Prediction App")
st.write("Upload a crop image to predict its height:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # 3. Preprocess image
    # -----------------------------
    # Resize to match your model input (change size if your model expects different)
    img_size = (224, 224)
    image_resized = image.resize(img_size)
    image_array = np.array(image_resized).astype(np.float32)

    # Normalize if your model requires (0-1 range)
    image_array = image_array / 255.0

    # Add batch dimension
    input_data = np.expand_dims(image_array, axis=0)  # Shape: [1, 224, 224, 3]

    # -----------------------------
    # 4. Predict with ONNX model
    # -----------------------------
    input_name = session.get_inputs()[0].name
    try:
        result = session.run(None, {input_name: input_data})
        predicted_height = result[0][0]  # adjust index if needed
        st.success(f"Predicted Crop Height: {predicted_height:.2f} units")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
