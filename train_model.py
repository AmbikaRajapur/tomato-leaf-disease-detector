import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import requests
from io import BytesIO

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------
# Load dataset
# ----------------------------
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
classes = train_dataset.classes

# ----------------------------
# CNN model
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
            nn.Linear(256,len(classes))
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(10):
    for images,labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch+1, "Loss:", loss.item())

# Save model
MODEL_PATH = "model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print("Training completed and model saved!")

# ----------------------------
# Prevention tips
# ----------------------------
prevention_tips = {
    "Early_Blight": "Remove infected leaves, avoid overhead watering, rotate crops, and apply fungicides if necessary.",
    "Late_Blight": "Use resistant varieties, avoid wet conditions, and apply fungicides early.",
    "Leaf_Mold": "Improve air circulation, avoid excessive nitrogen, and use fungicide sprays if needed.",
    "Septoria": "Remove infected leaves, rotate crops, and avoid wetting leaves.",
    "Healthy": "Plant is healthy. Continue regular care and monitoring."
}

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(image_path_or_url):
    """
    Input: image path or URL
    Output: disease, confidence, prevention
    """
    # Load image
    if str(image_path_or_url).startswith("http"):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_path_or_url).convert("RGB")
    
    # Preprocess
    transform_img = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img_tensor = transform_img(img).unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs,1)
        disease = classes[pred.item()]
        confidence = conf.item()
        prevention = prevention_tips.get(disease, "No info available.")
    
    return img, disease, confidence, prevention

# ----------------------------
# Example usage
# ----------------------------
# Replace with your uploaded image path or URL
test_image = "dataset/test/Early_Blight/sample1.jpg"  # or URL
img, disease, conf, prevention = predict_image(test_image)

# Show results
img.show(title="Input Image")
print("Detected Disease:", disease)
print("Confidence:", round(conf*100,2), "%")
print("How to prevent/manage:", prevention)