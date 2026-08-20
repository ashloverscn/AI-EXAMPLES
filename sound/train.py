import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf
import torchaudio.transforms as T

class AudioMFCCDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=3.0):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.file_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith(".wav"):
                    self.file_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        # Use MFCC instead of MelSpectrogram for speech tasks
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64}
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        data, sr = sf.read(audio_path)
        waveform = torch.tensor(data, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[1] < self.num_samples:
            pad_amount = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :self.num_samples]
            
        # Extract MFCC features [1, n_mfcc, time]
        mfcc = self.mfcc_transform(waveform)
        return mfcc, label

class MFCCAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(MFCCAudioCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def train_model():
    dataset_path = "dataset"
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        print("Error: No dataset folder found!")
        return

    full_dataset = AudioMFCCDataset(dataset_path)
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFCCAudioCNN(num_classes=len(full_dataset.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 40
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_audio_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n[Smart Stop] Early stopping activated. Model saved!")
                break

if __name__ == "__main__":
    train_model()