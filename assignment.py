import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

st.set_page_config(page_title="Digit Predictor", layout="centered")
st.title("✏️ Handwritten Digit Recognition")
st.write("Upload a 25x25 grayscale image, and the model will predict the digit.")

# --- Load MNIST and preprocess ---
@st.cache_data
def load_data():
    (train_images, train_labels), (_, _) = mnist.load_data()
    train_images_scaled = train_images.reshape((60000, 784)).astype("float32") / 255
    return train_images_scaled, train_labels

train_images_scaled, train_labels = load_data()

# --- Build & train model ---
@st.cache_resource
def build_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images_scaled, train_labels, epochs=5, batch_size=128, verbose=0)
    return model

model = build_model()

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a 25x25 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(img, caption="Uploaded Image", width=200)
    img_array = np.array(img)

    if img_array.shape != (25, 25):
        st.warning("⚠️ Image must be 25x25 pixels")
    else:
        # Resize to 28x28 for MNIST
        img_resized = img.resize((28,28))
        img_flat = np.array(img_resized).reshape(1,784).astype("float32") / 255

        # Predict
        prediction = model.predict(img_flat)
        st.success(f"Predicted Digit: {prediction.argmax()}")
