# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io

# Define constants
MODEL_SAVE_PATH = 'drink_classifier_model' # Assumes the model is in the same directory or accessible
IMG_SIZE = 128
class_names = ['Beer', 'Whiskey', 'Wine']

# Load the trained model
@st.cache_resource
def load_model():
    try:
        loaded_model_layer = tf.keras.layers.TFSMLayer(MODEL_SAVE_PATH, call_endpoint='serving_default')
        st.success(f"Model loaded successfully from: {MODEL_SAVE_PATH}")
        return loaded_model_layer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Please make sure 'drink_classifier_model' directory exists in the same location as 'app.py'.")
        st.stop()

model = load_model()

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    np_img = np.frombuffer(image_bytes, np.uint8)
    # Decode image to OpenCV format
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode image. Please upload a valid image file.")
        return None

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB (Streamlit and matplotlib usually expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img / 255.0
    # Add batch dimension
    img_for_prediction = np.expand_dims(img, axis=0)
    return img, img_for_prediction

# Streamlit application
st.title("Drink Classifier")
st.write("Upload an image of a drink to classify it as Beer, Whiskey, or Wine.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as bytes
    image_bytes = uploaded_file.getvalue()

    # Preprocess image
    display_img, img_for_prediction = preprocess_image(image_bytes)

    if img_for_prediction is not None:
        st.image(display_img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        raw_predictions = model(img_for_prediction)
        predictions = raw_predictions['output_0'].numpy()
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class_index]
        confidence_scores = predictions[0]

        st.subheader(f"Prediction: {predicted_label}")
        st.write(f"Confidence Scores:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {confidence_scores[i]*100:.2f}%")