%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from google.cloud import storage

# Define constants
# Model will be downloaded to a temporary directory
DOWNLOAD_DIR = '/tmp/downloaded_model'
LOCAL_MODEL_LOAD_PATH = os.path.join(DOWNLOAD_DIR, 'drink_classifier_model')

# This needs to be your GCS path where the model is stored
# Replace 'YOUR_GCS_BUCKET_NAME' and 'YOUR_GCS_MODEL_PATH' with actual values
# Example: GCS_MODEL_PATH = 'gs://your-bucket-name/drink_classifier_model'
GCS_MODEL_PATH = 'gs://drink_classifier_model_bucket_{YOUR_PROJECT_ID}/drink_classifier_model'
# IMPORTANT: Replace {YOUR_PROJECT_ID} with the actual Project ID you used above!

class_names = ['Beer', 'Whiskey', 'Wine'] # Ensure this matches your training labels
IMG_SIZE = 128 # Ensure this matches your training image size

st.title('Drink Classifier App')
st.write('Upload an image to classify it as Beer, Whiskey, or Wine.')

# Function to download the model from GCS
@st.cache_resource
def download_model_from_gcs(gcs_path, local_path):
    st.write(f"Attempting to download model from {gcs_path} to {local_path}")
    os.makedirs(local_path, exist_ok=True)
    
    bucket_name = gcs_path.split('//')[1].split('/')[0]
    model_blob_prefix = '/'.join(gcs_path.split('//')[1].split('/')[1:])
    
    client = storage.Client() # Assumes authentication is set up in the environment
    bucket = client.get_bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=model_blob_prefix)
    for blob in blobs:
        # Construct local file path, preserving directory structure
        local_file_path = os.path.join(local_path, os.path.relpath(blob.name, model_blob_prefix))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        st.write(f"Downloaded {blob.name} to {local_file_path}")
    st.success(f"Model downloaded to {local_path}")
    return local_path

# Load the trained model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model(model_path):
    try:
        loaded_model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        return loaded_model_layer
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}. Please ensure the model path is correct and the model is saved in the TensorFlow SavedModel format.")
        return None

# Download the model first
downloaded_model_path = download_model_from_gcs(GCS_MODEL_PATH, LOCAL_MODEL_LOAD_PATH)
model_layer = load_model(downloaded_model_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image bytes
    img_bytes = uploaded_file.getvalue()

    # Convert bytes to numpy array then to OpenCV format
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode image. Please upload a valid image file.")
    else:
        st.image(img_bytes, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized / 255.0  # Normalize pixel values
        img_for_prediction = np.expand_dims(img_normalized, axis=0) # Add batch dimension

        if model_layer:
            try:
                # Make a prediction using the loaded model layer
                raw_predictions = model_layer(img_for_prediction)
                # Access the output tensor from the dictionary
                predictions = raw_predictions['output_0'].numpy()

                predicted_class_index = np.argmax(predictions[0])
                predicted_label = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index]

                st.success(f"Prediction: **{predicted_label}** with {confidence*100:.2f}% confidence.")

                st.write("Confidences:")
                for i, class_name in enumerate(class_names):
                    st.write(f"- {class_name}: {predictions[0][i]*100:.2f}%")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Model not loaded. Cannot make predictions.")
