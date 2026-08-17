import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== CONFIGURATION ====================
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_SAVE_PATH = "cifar_cnn.pth"
CLASSES_JSON_PATH = "classes.json"
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 18
LEARNING_RATE = 0.001
PATIENCE = 3  # Overfitting cap / Early stopping patience
# =======================================================


# Define the CNN Architecture globally so other scripts can import it
class CIFARClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFARClassifierCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # 1. Device Configuration (GPU first, fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Data Augmentation and Normalization Pipeline
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # 3. Load Datasets
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(
            f"[ERROR] Dataset directories not found! Checked '{TRAIN_DIR}' and '{TEST_DIR}'. "
            "Please run your downloader script first."
        )
        return

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR, transform=transform_train
    )
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform_test)

    # Save classes to a JSON file automatically during training
    classes = train_dataset.classes
    with open(CLASSES_JSON_PATH, "w") as f:
        json.dump(classes, f, indent=4)
    print(f"[INFO] Classes saved to '{CLASSES_JSON_PATH}': {classes}")

    # Using num_workers=0 to prevent multiprocessing spawn errors on Windows
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"[INFO] Total training images: {len(train_dataset)}")
    print(f"[INFO] Total test images: {len(test_dataset)}")

    model = CIFARClassifierCNN(num_classes=len(classes)).to(device)

    # 4. Load Previous Weights if Available
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Loading existing model weights from '{MODEL_SAVE_PATH}'...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("[INFO] Previous weights loaded successfully. Continuing training...")
    else:
        print("[INFO] No previous weights found. Starting training from scratch.")

    # 5. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop with Overfitting Monitor (Early Stopping & Best Weights Saving)
    print("\n[INFO] Starting model training...")
    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = (correct_train / total_train) * 100

        # Validation on Test Set
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = (correct_test / total_test) * 100

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train "
            f"Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

        # Overfitting Cap Check: Save best model and check patience
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> [INFO] Accuracy improved. Model saved to '{MODEL_SAVE_PATH}'.")
        else:
            patience_counter += 1
            print(
                f"  -> [WARNING] No improvement. Patience counter: "
                f"{patience_counter}/{PATIENCE}"
            )
            if patience_counter >= PATIENCE:
                print(
                    f"\n[INFO] Overfitting cap reached! Early stopping triggered at "
                    f"epoch {epoch+1}."
                )
                break

    print(
        f"\n[INFO] Training complete! Best Test Accuracy achieved: "
        f"{best_test_acc:.2f}%"
    )


if __name__ == "__main__":
    main()