import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import os

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:\tara_dataset"
CUSTOM_PATH = r"D:\tara_dataset\custom"

BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    print("Starting full retraining...")
    print("Using device:", DEVICE)

    # =========================
    # TRANSFORMS
    # =========================

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # =========================
    # DATASETS
    # =========================

    train_dataset_main = datasets.ImageFolder(
        root=os.path.join(DATASET_PATH, "train"),
        transform=train_transforms
    )

    custom_dataset = datasets.ImageFolder(
        root=CUSTOM_PATH,
        transform=train_transforms
    )

    train_dataset = ConcatDataset([train_dataset_main, custom_dataset])

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_PATH, "test"),
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # =========================
    # MODEL
    # =========================

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler("cuda")

    best_val_acc = 0.0

    # =========================
    # TRAINING LOOP
    # =========================

    for epoch in range(EPOCHS):

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        model.train()
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if batch_idx % 200 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "tara_classifier.pth")
            print("✅ Improved model saved!")

    print("\nTraining complete!")
    print("Best Validation Accuracy:", best_val_acc)


# Windows multiprocessing fix
if __name__ == "__main__":
    main()
