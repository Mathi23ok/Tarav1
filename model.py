import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Model
# ===============================
model = models.efficientnet_b0(pretrained=False)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("tara_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# Prediction
# ===============================
def predict_ai_probability(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()[0]


# ===============================
# Embedding Extraction (Correct Way)
# ===============================
def extract_embedding(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.features(image_tensor)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        embedding = pooled.view(pooled.size(0), -1)

    return embedding.cpu().numpy()[0]


# ===============================
# REAL Grad-CAM Implementation
# ===============================
def generate_gradcam(image: Image.Image):

    image_tensor = transform(image).unsqueeze(0).to(device)
    image_np = np.array(image.resize((224, 224)))

    # Hook storage
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Use last convolutional block
    target_layer = model.features[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = torch.argmax(output, dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-8

    heatmap = heatmap.cpu().detach().numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    os.makedirs("explanations", exist_ok=True)
    path = f"explanations/{np.random.randint(1000000)}.png"
    cv2.imwrite(path, overlay)

    forward_handle.remove()
    backward_handle.remove()

    return path
