import streamlit as st
import requests
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from predict import predict_disease
from disease_info import disease_explanations

# Page configuration
st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# Custom UI style
st.markdown("""
<style>
.title{
font-size:40px;
font-weight:bold;
color:#2E8B57;
}
.card{
padding:20px;
border-radius:12px;
background-color:#f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🌿 Tomato Disease Detection Dashboard</p>', unsafe_allow_html=True)
st.write("Upload a tomato leaf image or paste an image URL to detect diseases using a CNN model.")

# Sidebar
st.sidebar.header("⚙ Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

url = st.sidebar.text_input("Or Enter Image URL")

image = None

# Load image
if uploaded_file:
    image = Image.open(uploaded_file)

if url:
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)

# Main dashboard
if image:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_column_width=True)

    image.save("temp.jpg")

    disease, confidence, probs = predict_disease("temp.jpg")

    with col2:

        st.subheader("🧠 AI Prediction")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.success(f"Detected Disease: **{disease}**")

        st.metric(
            label="Confidence Score",
            value=f"{round(confidence*100,2)}%"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Probability chart
    st.subheader("📊 Prediction Probability")

    df = pd.DataFrame({
        "Disease": list(probs.keys()),
        "Probability": list(probs.values())
    })

    fig, ax = plt.subplots()

    ax.bar(df["Disease"], df["Probability"])

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # AI Explanation
    st.subheader(" Explanation")

    info = disease_explanations[disease]

    st.write("###  Cause")
    st.write(info["cause"])

    st.write("###  Symptoms Detected")
    st.write(info["symptoms"])

    st.write("###  Recommended Solution")
    st.write(info["solution"])

else:

    st.info("Upload an image or enter an image URL to start detection.")