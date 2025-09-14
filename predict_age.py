import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser(description="Predict age from a dorsal hand image")
parser.add_argument("image_path", type=str, help="Path to the hand image")
parser.add_argument("--model_path", type=str, default="hand_age_model.pth", help="Path to the trained model weights")
args = parser.parse_args()

if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image not found: {args.image_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model weights not found: {args.model_path}")

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Model setup
# ---------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)  # Regression output
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
model = model.to(device)

# ---------------------------
# Image transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Prediction function
# ---------------------------
def predict_age(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        age_pred = model(image)
    return age_pred.item()

# ---------------------------
# Run prediction
# ---------------------------
predicted_age = predict_age(args.image_path)
print(f"Predicted age: {predicted_age:.2f} years")