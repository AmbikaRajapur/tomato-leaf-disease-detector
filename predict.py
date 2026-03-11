import torch
import gdown
import os

MODEL_PATH = "model.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/file/d/1-lXXV0iMHOEP-qaH3IptbKtcb5dffeG2/view?usp=sharing"

    gdown.download(url, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Classes
classes = [
"Early_Blight",
"Late_Blight",
"Leaf_Mold",
"Septoria",
"Healthy"
]

# CNN Model
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


# Load model
model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def predict_disease(img_path):

    img = Image.open(img_path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    output = model(img)

    probs = torch.softmax(output,1)[0]

    prob_dict = {}

    for i,cls in enumerate(classes):
        prob_dict[cls] = float(probs[i])

    confidence, pred = torch.max(probs,0)

    return classes[pred.item()], confidence.item(), prob_dict