import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()
st.write("MNIST dataset loaded.")

st.write("Here are some sample images from the MNIST dataset:")
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
for i in range(num_samples):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].axis('off')
st.pyplot(fig)

model = load_model('your_model.h5')
st.write("Model loaded successfully.")

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to the model's input shape
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    st.image(image[0, :, :, 0], caption="Uploaded Image", use_column_width=True)

    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    st.write(f"Predicted Digit: {predicted_digit}")
