import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os

def test_model(model_path, data_dir='./dataset/', batch_size=32):
    # Step 1: Set up data transformations for validation
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Step 2: Load validation dataset
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)

    dataset_sizes = {'val': len(val_dataset)}
    class_names = val_dataset.classes

    # Step 3: Set up device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Step 4: Load the pre-trained MobileNetV3-Small model structure
    model = models.mobilenet_v3_small(pretrained=False) # No need for pretrained weights here, we load our own
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    
    # Step 5: Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Set model to evaluate mode

    print(f"Testing model: {model_path}")
    print(f"Number of validation samples: {dataset_sizes['val']}")
    print(f"Classes: {class_names}")

    running_corrects = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Step 6: Evaluate the model
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']

    print(f'Validation Loss: {epoch_loss:.4f}')
    print(f'Validation Accuracy: {epoch_acc:.4f}')

if __name__ == '__main__':
    model_checkpoint_path = 'forest_fire_classifier_mobilenetv3_small_checkpoint.pth'
    test_model(model_checkpoint_path)
