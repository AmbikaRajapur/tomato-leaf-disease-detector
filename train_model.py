import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

classes = train_dataset.classes

# CNN model
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


model = CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):

    for images,labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:",epoch+1,"Loss:",loss.item())

# Save model
torch.save(model.state_dict(),"model.pth")

print("Training completed")