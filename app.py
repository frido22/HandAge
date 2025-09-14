from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)

# Download and load model from external source
import urllib.request
import urllib.error

def download_model():
    # Use Dropbox direct download URL (change dl=0 to dl=1)
    model_url = "https://www.dropbox.com/scl/fi/qjgnejdirb3y9i78mbrgs/hand_age_model.pth?rlkey=rno02w0f1pjw7lxrg1a7p64vb&st=auesaza9&dl=1"
    model_path = "hand_age_model.pth"
    
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
            return True
        except urllib.error.URLError as e:
            print(f"Failed to download model: {e}")
            return False
    else:
        print("Model already exists locally")
        return True

# Load trained model
if download_model():
    try:
        checkpoint = torch.load("hand_age_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        print("Loaded trained model")
        del checkpoint
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Warning: Using untrained model.")
else:
    print("Warning: Using untrained model - model download failed.")

model.eval()
model = model.to(device)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_age_from_image(image):
    """Predict age from PIL Image"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            age_pred = model(image_tensor)
        
        return float(age_pred.item())
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})
        
        # Remove data URL prefix if present
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict age
        predicted_age = predict_age_from_image(image)
        
        # Ensure reasonable age range (clamp between 0-100)
        predicted_age = max(0, min(100, predicted_age))
        
        return jsonify({
            'success': True, 
            'age': predicted_age
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Error processing image: {str(e)}'
        })

if __name__ == '__main__':
    print("Starting Hand Age Predictor Web App...")
    print(f"Model loaded on device: {device}")
    
    # Use environment variables for production deployment
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    if debug:
        print(f"Open http://localhost:{port} in your browser")
    
    app.run(debug=debug, host='0.0.0.0', port=port)