import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    epochs = 18
    learning_rate = 0.001

    dataset_path = "./dataset/train"
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset directory '{dataset_path}' not found! Please run your extraction script first.")
        return

    # --- 3. Setup Image Transforms & Load from Folders ---
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean and standard deviation
    ])

    print("Loading image dataset from folders...")
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Classes found: {train_dataset.classes}")
    print(f"Total training images loaded: {len(train_dataset)}")

    # --- 4. Initialize Enhanced Model, Loss, and Optimizer ---
    model = EnhancedResCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. Training Loop ---
    print(f"\nStarting training with Enhanced ResNet Architecture for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_acc = 100. * correct / total
        print(f"--- Epoch {epoch+1} Complete | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}% ---")

    # --- 6. Save Model Weights ---
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("\nTraining complete! Enhanced model weights successfully saved as 'mnist_cnn.pth'.")

if __name__ == '__main__':
    main()
