
import torch
import torch.quantization
import torchvision.models as models

# Step 1: Load the MobileNetV3-Small model
model = models.mobilenet_v3_small(pretrained=False)

# Step 2: Modify the final classifier layer to match our two classes
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

# Step 3: Load the trained weights
state_dict = torch.load('forest_fire_classifier_mobilenetv3_small.pth')
model.load_state_dict(state_dict)

# Step 4: Set the model to evaluation mode
model.eval()

# Step 5: Apply dynamic quantization to reduce model size and improve CPU speed
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Step 6: Create an example input tensor for tracing
example_input = torch.randn(1, 3, 224, 224)

# Step 7: Trace the quantized model
traced_script_module = torch.jit.trace(model, example_input)

# Step 8: Save the quantized TorchScript model for deployment
traced_script_module.save('model_int8.pt')
