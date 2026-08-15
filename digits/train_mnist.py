import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- 1. Define Enhanced Residual Block & Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancedResCNN(nn.Module):
    def __init__(self):
        super(EnhancedResCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.res_block1 = ResidualBlock(32)
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.res_block2 = ResidualBlock(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.mid_conv(x)
        x = self.res_block2(x)
        
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    # --- 2. Setup Device & Hyperparameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 64
    epochs = 30
    learning_rate = 0.0005

    dataset_path = "./dataset/train"
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset directory '{dataset_path}' not found!")
        return

    # --- 3. Setup Image Transforms & Load Dataset ---
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {train_size} | Validation samples: {val_size}")

    # --- 4. Initialize Model, Weights, Loss, and Optimizer ---
    model = EnhancedResCNN().to(device)
    
    MODEL_PATH = "mnist_cnn.pth"
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing weights from '{MODEL_PATH}'...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. Early Stopping Variables ---
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print(f"\nStarting training with Early Stopping protection...\n")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Print status every 200 batches so you know it's actively working
            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Current Batch Loss: {loss.item():.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        print(f"\n--- EPOCH {epoch+1} FINISHED ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

        # --- Overfitting / Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  [-] Validation loss improved. Saving best checkpoint.\n")
        else:
            patience_counter += 1
            print(f"  [!] No validation improvement ({patience_counter}/{patience}).\n")
            if patience_counter >= patience:
                print("\nEarly stopping triggered! Overfitting protection cap reached. Halting training.")
                break

    print(f"\nTraining complete. Best model weights saved to '{MODEL_PATH}'.")

if __name__ == '__main__':
    main()
