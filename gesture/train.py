import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== CONFIGURATION ====================
TRAIN_DATASET_DIR = "dataset/train"
MODEL_SAVE_PATH = "face_cnn_weights.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 50                # Absolute upper limit
TARGET_ACCURACY = 99.5         # Overfitting cap threshold (%)
LEARNING_RATE = 0.001
# =======================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(TRAIN_DATASET_DIR):
        print(f"[ERROR] Training directory '{TRAIN_DATASET_DIR}' not found!")
        return

    # 1. Balanced Data Augmentation Pipeline
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DATASET_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"[INFO] Found {len(train_dataset)} training images across {num_classes} classes: {classes}")

    if num_classes == 0:
        print("[ERROR] No classes or images found in dataset/train!")
        return

    # 2. Balanced 3-Block CNN Architecture
    class BalancedSignLanguageCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = BalancedSignLanguageCNN(num_classes=num_classes).to(device)

    # 3. Load Previous Weights if they exist
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            print(f"[INFO] Existing weights found at '{MODEL_SAVE_PATH}'. Loading weights to resume training...")
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[INFO] Weights loaded successfully!")
        except Exception as e:
            print(f"[WARNING] Could not load previous weights ({e}). Training from scratch.")
    else:
        print("[INFO] No existing weight file found. Starting fresh training from scratch.")

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 5. Continuous Training Loop until Overfitting Cap
    print(f"\n[INFO] Training until reaching target accuracy cap of {TARGET_ACCURACY}% (Max {MAX_EPOCHS} epochs)...")
    
    epoch = 0
    while epoch < MAX_EPOCHS:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct / total) * 100

        print(f"Epoch [{epoch}/{MAX_EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # Check if we hit the overfitting threshold cap
        if epoch_acc >= TARGET_ACCURACY:
            print(f"\n[INFO] Target accuracy cap of {TARGET_ACCURACY}% reached at epoch {epoch}! Stopping training.")
            break

    # 6. Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classes': classes
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"\n[INFO] Model successfully trained and saved to '{MODEL_SAVE_PATH}'!")

if __name__ == '__main__':
    main()