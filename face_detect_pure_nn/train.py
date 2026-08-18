import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== CONFIGURATION ====================
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "face_cnn_weights.pth"
IMG_SIZE = 128
BATCH_SIZE = 4
MAX_EPOCHS = 100  # Set a high ceiling since we are using over-fitting cap
LEARNING_RATE = 0.001

# Overfitting / Stopping Cap Parameters
OVERFIT_LOSS_THRESHOLD = 0.01  # Stop when training loss is extremely low
OVERFIT_ACC_THRESHOLD = 100.0   # Stop when training accuracy hits 100%
PATIENCE = 3                    # Number of consecutive epochs to sustain top accuracy/low loss before stopping
# =======================================================

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. Data Augmentation and Normalization Pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
    print(f"[ERROR] Dataset directory '{DATASET_DIR}' is missing or empty!")
    exit(1)

dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

classes = dataset.classes
print(f"[INFO] Found classes: {classes}")
print(f"[INFO] Total training images: {len(dataset)}")

# 4. Define the CNN Architecture
class FaceClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifierCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = FaceClassifierCNN(num_classes=len(classes)).to(device)

# 4.1 Preload Model Weights if Present
if os.path.exists(MODEL_SAVE_PATH):
    print(f"[INFO] Found existing weights at '{MODEL_SAVE_PATH}'. Loading weights...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    
    # Optional check to ensure class mismatch doesn't break things
    if 'classes' in checkpoint and checkpoint['classes'] != classes:
        print("[WARNING] Saved classes do not match current dataset classes! Proceeding with caution.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[INFO] Successfully preloaded model weights.")
else:
    print("[INFO] No existing weight file found. Training from scratch.")

# 5. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training Loop with Overfitting Cap Detection
print("\n[INFO] Starting model training...")
model.train()

overfit_counter = int(0)

for epoch in range(MAX_EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataset)
    epoch_acc = (correct / total) * 100
    
    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Check for Overfitting Cap condition (e.g., 100% accuracy and near-zero loss)
    if epoch_acc >= OVERFIT_ACC_THRESHOLD and epoch_loss <= OVERFIT_LOSS_THRESHOLD:
        overfit_counter += 1
        print(f"[INFO] Overfitting cap indicators reached ({overfit_counter}/{PATIENCE}).")
        if overfit_counter >= PATIENCE:
            print(f"\n[INFO] Overfitting cap threshold reached consistently. Stopping training early.")
            break
    else:
        overfit_counter = 0  # Reset if it fluctuates back

# 7. Save Model Weights and Classes Dictionary
checkpoint = {
    'model_state_dict': model.state_dict(),
    'classes': classes
}
torch.load_state_dict = model.state_dict() # safe reference
torch.save(checkpoint, MODEL_SAVE_PATH)
print(f"\n[INFO] Training complete! Model weights successfully saved to '{MODEL_SAVE_PATH}'.")