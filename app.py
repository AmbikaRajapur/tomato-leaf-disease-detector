import streamlit as st
import requests
from PIL import Image
import tempfile
import os
from predict import predict_disease

st.title("AI Tomato Leaf Disease Detector")

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# URL input
url = st.text_input("Or Enter Image URL")

image = None

if uploaded_file:
    image = Image.open(uploaded_file)

if url:
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)

if image:

    st.image(image, caption="Input Image")

    temp_path = "temp.jpg"
    image.save(temp_path)

    disease, conf = predict_disease(temp_path)

    st.subheader("Prediction")

    st.write("Result:", disease)
    st.write("Confidence:", round(conf*100,2), "%")

    # Label correction
    label = st.selectbox(
        "Correct label if prediction is wrong",
        ["Early_Blight","Late_Blight","Leaf_Mold","Septoria","Healthy"]
    )

    if st.button("Add to Dataset & Retrain"):

        save_folder = f"dataset/train/{label}"

        os.makedirs(save_folder, exist_ok=True)

        image.save(os.path.join(save_folder, "new_image.jpg"))

        st.success("Image added to dataset!")

        os.system("python train_model.py")

        st.success("Model retrained successfully!")