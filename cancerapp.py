import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

# Load the trained model
model = load_model('Cancer_data.h5')

# Define class indices
class_indices = {0: 'cancer', 1: 'not Cancer'}

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to the input shape your model expects
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize if your model expects it
    return image

# Define the Streamlit app
st.title('Cancer Classifier App')

st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    progress_bar = st.progress(0)
    
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent_complete)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    st.write(prediction)
    
    # Interpret the result based on the 0.5 threshold
    if prediction[0][0] > 0.5:
        st.info("You are not  Patient Of Cancer")
    else:
        st.warning("You are patient of cancer")
