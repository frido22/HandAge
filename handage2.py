import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

# ---------------------------
# Model setup
# ---------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Replace last FC layer for regression (age prediction)
model.fc = nn.Linear(model.fc.in_features, 1)

# ---------------------------
# Custom Dataset
# ---------------------------
class HandAgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Load CSV with headers
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        valid_rows = []
        for idx in range(len(self.data)):
            aspect = str(self.data.iloc[idx]["aspectOfHand"]).lower()
            img_name = str(self.data.iloc[idx]["imageName"])
            img_path = os.path.join(self.img_dir, img_name)

            if "dorsal" in aspect and os.path.exists(img_path):
                valid_rows.append(idx)

        self.valid_indices = valid_rows

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_name = str(self.data.iloc[real_idx]["imageName"])
        age = self.data.iloc[real_idx]["age"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)

# ---------------------------
# Image transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# Dataset and DataLoader
# ---------------------------
dataset = HandAgeDataset(csv_file="HandInfo.csv", img_dir="Hands", transform=transform)

print("Total dorsal images found:", len(dataset))
if len(dataset) == 0:
    raise RuntimeError("No dorsal images found. Check CSV column names and img_dir path.")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------------------------
# Loss and optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(5):  # Adjust epochs as needed
    model.train()
    running_loss = 0.0
    for images, ages in dataloader:
        images, ages = images.to(device), ages.to(device).unsqueeze(1)  # ensure shape [B,1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# ---------------------------
# Prediction function
# ---------------------------
def predict_age(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        age_pred = model(image)
    return age_pred.item()

# After the training loop
torch.save(model.state_dict(), "hand_age_model.pth")
print("Model saved successfully!")

# ---------------------------
# Example usage
# ---------------------------
# predicted = predict_age("Hands/example.jpg")
# print("Predicted age:", predicted)
