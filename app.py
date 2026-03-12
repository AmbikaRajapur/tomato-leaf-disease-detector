import streamlit as st
from PIL import Image
import tempfile
import os
from predict import predict_disease, MODEL_PATH

st.title("🍅 AI Tomato Leaf Disease Detector")

# ----------------------------
# Upload or URL input
# ----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
url = st.text_input("Or Enter Image URL")

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif url:
    image = url  # Will be handled by predict_disease

# ----------------------------
# Show prediction
# ----------------------------
if image:
    if isinstance(image, Image.Image):
        st.image(image, caption="Input Image", use_column_width=True)
        # Save temp file for prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            img, disease, conf, prevention = predict_disease(tmp.name)
    else:
        # image is URL
        img, disease, conf, prevention = predict_disease(image)
        st.image(img, caption="Input Image", use_column_width=True)
    
    st.subheader("Prediction")
    st.markdown(f"**Detected Disease:** {disease}")
    st.markdown(f"**Confidence:** {round(conf*100,2)} %")

    st.subheader("How to prevent / manage this disease:")
    st.markdown(prevention)

else:
    st.info("Upload an image or enter an image URL to detect disease.")