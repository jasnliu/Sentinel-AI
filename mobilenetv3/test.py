import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import argparse


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path

def test_model(model_path, data_dir="./dataset/", batch_size=32, output_path="result.log"):
    # Step 1: Set up data transformations for validation
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Step 2: Load validation dataset
    val_dataset = ImageFolderWithPaths(os.path.join(data_dir, "val"), data_transforms)
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
    with open(output_path, "w", encoding="utf-8") as log_f, torch.no_grad():
        log_f.write("path\tpred\ttrue\tconfidence\tcorrect\n")
        for inputs, labels, paths in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            preds_cpu = preds.detach().cpu().tolist()
            labels_cpu = labels.detach().cpu().tolist()
            confs_cpu = confs.detach().cpu().tolist()

            for path, pred_idx, true_idx, conf in zip(paths, preds_cpu, labels_cpu, confs_cpu):
                pred_name = class_names[pred_idx]
                true_name = class_names[true_idx]
                correct = int(pred_idx == true_idx)
                log_f.write(f"{path}\t{pred_name}\t{true_name}\t{conf:.6f}\t{correct}\n")

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']

    print(f'Validation Loss: {epoch_loss:.4f}')
    print(f'Validation Accuracy: {epoch_acc:.4f}')
    print(f"Wrote per-file results to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3-Small forest fire classifier.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .pth file (default: forest_fire_classifier_mobilenetv3_small.pth if present, else checkpoint).",
    )
    parser.add_argument("--data-dir", default="./dataset/", help="Dataset root containing val/ (default: ./dataset/).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation (default: 32).")
    parser.add_argument("--output", default="result.log", help="Output log file path (default: result.log).")
    args = parser.parse_args()

    default_model = (
        "forest_fire_classifier_mobilenetv3_small.pth"
        if os.path.exists("forest_fire_classifier_mobilenetv3_small.pth")
        else "forest_fire_classifier_mobilenetv3_small_checkpoint.pth"
    )
    model_path = args.model or default_model
    test_model(model_path, data_dir=args.data_dir, batch_size=args.batch_size, output_path=args.output)
