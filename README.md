# 🧠 Image Classification App

A simple Streamlit app that uses a trained deep learning model to classify images from the CIFAR-10 dataset.  
You can upload your own image, and the app will predict which class it belongs to (e.g., cat, dog, airplane, etc.).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployment-url.streamlit.app/)

---

### 🚀 Features
- Upload image and get prediction instantly
- Displays predicted class and confidence
- Visualizes probabilities for all classes as a bar chart in percentages

---

### 🛠 How to run it locally

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt

2. **Make sure you have the model file**
Place your trained model file in the project directory.

2. **Run the app**

   ```bash
   streamlit run streamlit_app.py
