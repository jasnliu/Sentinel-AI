import torch
import torch.quantization
import torch.nn as nn
import sys
from torchvision import models, transforms
from PIL import Image

# Ensure that an image path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python rec.py <image_path>")
    sys.exit(1)

# Path to the saved model and the test image from the command line
model_path = 'forest_fire_classifier_mobilenetv3_small.pth'
image_path = sys.argv[1]  # Get the first argument passed from the command line

# Step 1: Load the trained MobileNetV3-Small model
model_ft = models.mobilenet_v3_small(pretrained=False)
model_ft.classifier[3] = nn.Linear(model_ft.classifier[3].in_features, 2)

# Load the trained weights
model_ft.load_state_dict(torch.load(model_path))
model_ft.eval()
# Apply dynamic quantization to reduce model size and improve CPU speed
model_ft = torch.quantization.quantize_dynamic(model_ft, {nn.Linear}, dtype=torch.qint8)
# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Step 2: Define the preprocessing function (same as during training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Step 3: Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Step 4: Perform inference
def predict(image_path):
    # Preprocess the image
    input_image = preprocess_image(image_path)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model_ft(input_image)
    
    # Get the predicted class (0: fire, 1: no_fire)
    _, pred = torch.max(output, 1)
    
    # Return the prediction
    return 'no_fire' if pred.item() == 1 else 'fire'

# Step 5: Predict the class for the test image
predicted_class = predict(image_path)
print(f'The image is predicted to be a {predicted_class}.')

