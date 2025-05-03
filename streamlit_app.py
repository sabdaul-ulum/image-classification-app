import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf

# CIFAR-10 labels
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
               "Dog", "Frog", "Horse", "Ship", "Truck"]

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/cifar10_best_model.keras")

model = load_model()

# Judul aplikasi
st.title("Image Classification Apps")

st.write("Upload sebuah gambar untuk melihat prediksi dari model CNN menggunakan dataset CIFAR-10.")

# Upload file
uploaded_file = st.file_uploader("Silakan upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Tampilkan gambar
    st.image(image, caption="Gambar Input", use_container_width=True)

    # Preprocessing
    image_resized = image.convert("RGB").resize((32, 32))  # CIFAR-10 input size
    image_array = np.array(image_resized) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambah batch dimensi

    # Prediksi
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"Label: **{predicted_class}**")
    st.write(f"Akurasi: **{confidence * 100:.2f}%**")

    st.subheader("Probabilitas Kelas:")
    for class_name, prob in zip(class_names, prediction[0]):
        percent = prob * 100
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**{class_name}**")
            st.progress(float(prob))  # konversi ke float Python
        with col2:
            st.markdown(f"**{percent:.0f}%**")
