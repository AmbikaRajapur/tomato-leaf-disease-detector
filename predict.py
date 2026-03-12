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

# Load image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif url:
    try:
        response = requests.get(url, stream=True)
        image = Image.open(response.raw).convert("RGB")
    except:
        st.error("Failed to load image from URL")

if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Temporary file for prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        disease, conf = predict_disease(tmp.name)

    st.subheader("Prediction")
    st.write("Result:", disease)
    st.write("Confidence:", round(conf*100, 2), "%")

    # Correct label option
    label = st.selectbox(
        "Correct label if prediction is wrong",
        ["Early_Blight","Late_Blight","Leaf_Mold","Septoria","Healthy"]
    )

    if st.button("Add to Dataset & Retrain"):
        save_folder = f"dataset/train/{label}"
        os.makedirs(save_folder, exist_ok=True)
        image.save(os.path.join(save_folder, "new_image.jpg"))

        st.success("Image added to dataset!")

        # Retraining warning
        try:
            os.system("python train_model.py")
            st.success("Model retrained successfully!")
        except:
            st.warning("Retraining may not work on Streamlit Cloud. Try retraining locally.")
