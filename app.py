%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Assuming MODEL_SAVE_PATH and class_names are known or defined in the app
MODEL_SAVE_PATH = 'drink_classifier_model' # Updated to relative path
class_names = ['Beer', 'Whiskey', 'Wine'] # Ensure this matches your training labels
IMG_SIZE = 128 # Ensure this matches your training image size

st.title('Drink Classifier App')
st.write('Upload an image to classify it as Beer, Whiskey, or Wine.')

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    try:
        # Load the SavedModel as a TFSMLayer
        loaded_model_layer = tf.keras.layers.TFSMLayer(MODEL_SAVE_PATH, call_endpoint='serving_default')
        return loaded_model_layer
    except Exception as e:
        st.error(f"Error loading model from {MODEL_SAVE_PATH}: {e}. Please ensure the model path is correct and the model is saved in the TensorFlow SavedModel format.")
        return None

model_layer = load_model()

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
