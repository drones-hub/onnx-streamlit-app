import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
from PIL import Image

# Direct download link (string)
model_url = "https://huggingface.co/bhagyashrideshmukh72/yieldprediction/resolve/main/NEW_trainedCropHeightModel_Customized.onnx"

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
