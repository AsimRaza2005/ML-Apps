import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('your_data.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((img_height, img_width))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title('Gender Recognizer')


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = image.resize((28, 28))  # Resize to the model's input shape
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    
    # Preprocess and predict
    
    img_height = 256  # Adjust this value as needeed
    img_width = 256

    #processed_image = preprocess_image(image)
    prediction = model.predict(image)
    
    # Assuming output layer has one neuron with sigmoid activation
    gender = 'Female' if prediction[0][0] > 0.5 else 'Male'
    # confidence = (1 - prediction[0][0]) if gender == 'Female' else prediction[0][0]
    
    st.write(f"Prediction: **{gender}**")
    # st.write(f"Confidence: **{confidence:.2f}**")

# Run the app
