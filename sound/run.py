import time
import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd

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

SAMPLE_RATE = 16000
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
CLASSES = ['ashish', 'dharam']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFCCAudioCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load("best_audio_model.pth", map_location=device))
    model.eval()

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64}
    )

    print("\n🎙️ MFCC LIVE VOICE CLASSIFIER READY. Speak now...")
    try:
        while True:
            audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            waveform = torch.tensor(audio_data, dtype=torch.float32).t()
            if waveform.shape[1] < NUM_SAMPLES:
                waveform = torch.nn.functional.pad(waveform, (0, NUM_SAMPLES - waveform.shape[1]))
            else:
                waveform = waveform[:, :NUM_SAMPLES]
                
            mfcc_spec = mfcc_transform(waveform).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(mfcc_spec)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
            print(f"👉 Prediction: ** {CLASSES[pred_idx.item()].upper()} ** ({conf.item()*100:.2f}%)")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nExited.")

if __name__ == "__main__":
    main()