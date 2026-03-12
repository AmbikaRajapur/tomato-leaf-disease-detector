import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import os

# ----------------------------
# Classes and Prevention Tips
# ----------------------------
classes = [
    "Early_Blight",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria",
    "Healthy"
]

prevention_tips = {
    "Early_Blight": "Remove infected leaves, avoid overhead watering, rotate crops, and apply fungicides if necessary.",
    "Late_Blight": "Use resistant varieties, avoid wet conditions, and apply fungicides early.",
    "Leaf_Mold": "Improve air circulation, avoid excessive nitrogen, and use fungicide sprays if needed.",
    "Septoria": "Remove infected leaves, rotate crops, and avoid wetting leaves.",
    "Healthy": "Plant is healthy. Continue regular care and monitoring."
}

# ----------------------------
# CNN Model Definition
# ----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*26*26,256),
            nn.ReLU(),
            nn.Linear(256,5)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------------
# Load Model (auto-create if missing)
# ----------------------------
MODEL_PATH = "model.pth"

def create_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("model.pth not found. Creating dummy model...")
        temp_model = CNNModel()
        # initialize random weights
        for param in temp_model.parameters():
            param.data.uniform_(0, 0.01)
        torch.save(temp_model.state_dict(), MODEL_PATH)
        print("model.pth created successfully.")

create_model_if_missing()

model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------
# Prediction Function
# ----------------------------
def predict_disease(image_input):
    """
    image_input: image path or URL
    returns: PIL image, disease_name, confidence, prevention_tip
    """
    # Load image
    if str(image_input).startswith("http"):
        response = requests.get(image_input)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")
    
    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        disease_name = classes[pred.item()]
        confidence = conf.item()
        prevention = prevention_tips.get(disease_name, "No info available.")
    
    return img, disease_name, confidence, prevention
