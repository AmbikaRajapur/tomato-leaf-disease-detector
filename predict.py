import torch
import torch.nn as nn
import os
import gdown
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# Define CNN model
class TomatoCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TomatoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = TomatoCNN(num_classes=5)

# Model path & URL (replace with your own)
MODEL_PATH = "model.pth"
MODEL_URL = "YOUR_MODEL_FILE_URL"  # e.g., Google Drive or HuggingFace link

# Download model if missing
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Labels
labels = ["Early_Blight","Late_Blight","Leaf_Mold","Septoria","Healthy"]

# Prediction function
def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return labels[pred.item()], conf.item()
