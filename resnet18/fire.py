#!/usr/bin/env python3

import sys
import torch
from PIL import Image
from torchvision import transforms

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python fire.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]

    # Load the quantized TorchScript model
    model = torch.jit.load('model_int8.pt')
    model.eval()

    # Define the image transformations (adjust as per your training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize using the mean and std used during training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Replace with your mean
                             std=[0.229, 0.224, 0.225])   # Replace with your std
    ])

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Move input and model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_batch = input_batch.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Define your class names
    class_names = ['fire', 'no fire']  # Replace with your actual class names

    # Print the confidence scores for all classes
    print("Confidence scores for all classes:")
    for idx, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"Class {idx}: {class_name}, Confidence: {prob.item():.4f}")

    # Get the predicted class
    predicted_class_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_idx].item()

    if predicted_class_idx < len(class_names):
        predicted_class_name = class_names[predicted_class_idx]
    else:
        predicted_class_name = f"Class index {predicted_class_idx}"

    # Print the predicted class
    print(f"\nPredicted class: {predicted_class_name} (index: {predicted_class_idx})")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
