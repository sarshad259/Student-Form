import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config("Digit Predictor", layout="centered")
st.title("🧠 Digit Recognition")
st.markdown("### Draw or upload a digit (0–9) and let the AI guess!")

# ------------------ LOAD & TRAIN MODEL ------------------
@st.cache_resource
def load_model():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 784)).astype("float32") / 255
    test_images = test_images.reshape((10000, 784)).astype("float32") / 255

    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=3, batch_size=128, verbose=0)
    return model

model = load_model()

# ------------------ INPUT OPTIONS ------------------
st.write("#### Choose how to input your digit:")

option = st.radio("Input Method", ["Draw Digit", "Upload Image"])

# ------------------ DRAWING OPTION ------------------
if option == "Draw Digit":
    from streamlit_drawable_canvas import st_canvas

    st.write("Draw a digit below 👇")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas.image_data is not None:
        img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))
        img = ImageOps.grayscale(img)
        img_array = np.array(img).reshape(1, 784).astype("float32") / 255

        if st.button("Predict Digit"):
            prediction = model.predict(img_array)
            pred_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"🧩 Predicted Digit: *{pred_digit}*")
            st.info(f"Confidence: {confidence:.2f}%")
            st.image(img.resize((100, 100)), caption="Processed Input", use_column_width=False)

# ------------------ UPLOAD OPTION ------------------
elif option == "Upload Image":
    uploaded = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("L").resize((28, 28))
        st.image(img, caption="Uploaded Image", width=150)

        img_array = np.array(img).reshape(1, 784).astype("float32") / 255

        if st.button("Predict Digit"):
            prediction = model.predict(img_array)
            pred_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"🧩 Predicted Digit: *{pred_digit}*")
            st.info(f"Confidence: {confidence:.2f}%")
