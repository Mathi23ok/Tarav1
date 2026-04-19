import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import random

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:\tara_dataset"
MODEL_PATH = "tara_classifier.pth"
BATCH_SIZE = 32
USE_SUBSET = True        # Set False to use full 20k
SUBSET_SIZE = 5000       # Speed mode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
val_dataset_full = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=val_transforms
)

if USE_SUBSET:
    subset_indices = random.sample(
        range(len(val_dataset_full)),
        min(SUBSET_SIZE, len(val_dataset_full))
    )
    val_dataset = Subset(val_dataset_full, subset_indices)
else:
    val_dataset = val_dataset_full

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Validation samples:", len(val_dataset))

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================
# EVALUATION
# =========================
all_probs = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):

        if batch_idx % 20 == 0:
            print(f"Processing batch {batch_idx}/{len(val_loader)}")

        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        fake_probs = probs[:, 0]  # FAKE class index

        all_probs.extend(fake_probs.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nEvaluation complete.")

# =========================
# ROC
# =========================
fpr, tpr, thresholds = roc_curve(all_labels, all_probs,pos_label=0)
roc_auc = auc(fpr, tpr)

print("AUC Score:", roc_auc)

# =========================
# SAVE ROC CURVE
# =========================
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")
