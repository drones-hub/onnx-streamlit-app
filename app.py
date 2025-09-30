import streamlit as st
import numpy as np
import onnxruntime as ort

# Title of the app
st.title("ONNX Model Prediction App")

# Replace generic inputs with actual features from your dataset

st.write("Enter UAV and field attributes for prediction:")

crop_age = st.selectbox("Crop Age (months)", [8, 10])
visual_category = st.selectbox("Visual Category", ["Sugarcane", "Non-sugarcane", "Mixed"])
yield_class = st.selectbox("Yield Class", ["Poor", "Good", "Very Good"])

# Encode categorical values as numbers for the ONNX model
visual_map = {"Sugarcane": 0, "Non-sugarcane": 1, "Mixed": 2}
yield_map = {"Poor": 0, "Good": 1, "Very Good": 2}

input_data = np.array([[
    crop_age,
    visual_map[visual_category],
    yield_map[yield_class],
]], dtype=np.float32)


# Prepare input data for ONNX model
input_data = np.array([[input1, input2, input3]], dtype=np.float32)

# Load ONNX model
@st.cache_data
def load_model():
    session = ort.InferenceSession("model.onnx")
    return session

session = load_model()

# Make prediction
if st.button("Predict"):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})
    st.success(f"Prediction: {result[0][0]}")
