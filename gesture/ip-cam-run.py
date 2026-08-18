import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
MODEL_PATH = "face_cnn_weights.pth"
IMG_SIZE = 128  # Matches training image size
BOX_SIZE = 300
CONFIDENCE_THRESHOLD = 60.0  # Minimum percentage to show green box
# =======================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found at '{MODEL_PATH}'. Please run train.py first.")
        return

    # 1. Load Model Checkpoint & Classes
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        classes = checkpoint['classes']
        num_classes = len(classes)
    except Exception as e:
        print(f"[ERROR] Failed to load model checkpoint: {e}")
        return

    # 2. Define the Balanced CNN Architecture (Exact match to train.py)
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] Balanced model loaded successfully! Classes recognized: {classes}")

    # 3. Validation Transform Pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"[ERROR] Could not open RTSP stream at {RTSP_URL}")
        return

    print("\n[INFO] Starting live evaluation stream. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame dropped. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        h, w, _ = frame.shape
        x1, y1 = (w - BOX_SIZE) // 2, (h - BOX_SIZE) // 2
        
        # Crop Region of Interest (ROI)
        roi = frame[y1:y1+BOX_SIZE, x1:x1+BOX_SIZE]

        if roi.size != 0:
            # Step 1: Convert live crop to Grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Map grayscale back to 3-channel structure expected by the model
            gray_3ch = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)

            # Step 3: Convert to tensor and send to device
            pil_img = Image.fromarray(gray_3ch)
            tensor_img = transform(pil_img).unsqueeze(0).to(device)

            # Step 4: Inference
            with torch.no_grad():
                outputs = model(tensor_img)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_char = classes[predicted_idx.item()]
                conf_score = confidence.item() * 100

            # Debug output to track live scores in your terminal
            print(f"Predicted: {predicted_char:<5} | Confidence: {conf_score:.2f}%")

            # Draw visual indicator on screen
            color = (0, 255, 0) if conf_score >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + BOX_SIZE, y1 + BOX_SIZE), color, 2)
            
            label_text = f"Sign: {predicted_char} ({conf_score:.1f}%)"
            cv2.putText(frame, label_text, (x1, max(30, y1 - 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Balanced CNN Gesture Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()