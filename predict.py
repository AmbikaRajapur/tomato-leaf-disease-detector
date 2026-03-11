import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# -----------------------------
# CNN MODEL
# -----------------------------
class TomatoCNN(nn.Module):
    def __init__(self):
        super(TomatoCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*26*26,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# -----------------------------
# LOAD MODEL
# -----------------------------

model = TomatoCNN()

MODEL_PATH = "model.pth"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/file/d/1-lXXV0iMHOEP-qaH3IptbKtcb5dffeG2/view?usp=sharing"

    gdown.download(url, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model.eval()


# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -----------------------------
# CLASSES
# -----------------------------

classes = [
    "Early_Blight",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria",
    "Healthy"
]


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_disease(img_path):

    img = Image.open(img_path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = torch.nn.functional.softmax(output, dim=1)

        confidence, predicted = torch.max(probs,1)

    disease = classes[predicted.item()]

    confidence = confidence.item()

    prob_dict = {}

    for i,c in enumerate(classes):
        prob_dict[c] = float(probs[0][i])


    return disease, confidence, prob_dict
