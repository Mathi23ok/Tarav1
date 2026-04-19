import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:\tara_dataset"
MODEL_PATH = "tara_classifier.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=val_transforms
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Validation samples:", len(val_dataset))

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

# =========================
# EVALUATION
# =========================
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))
