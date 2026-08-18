import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
MODEL_WEIGHTS_PATH = "face_cnn_weights.pth"
IMG_SIZE = 128
DETECTION_THRESHOLD = 0.90   # MTCNN face detection confidence
RECOGNITION_THRESHOLD = 0.75 # Classifier confidence to identify person, else '?'
CACHE_REFRESH_INTERVAL = 30  # Re-run detector & classifier every N frames to update cache
# =======================================================

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. Load Trained PyTorch Model & Classes
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"[ERROR] Trained weights '{MODEL_WEIGHTS_PATH}' not found! Please run 'train.py' first.")
    exit(1)

checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
classes = checkpoint['classes']
print(f"[INFO] Loaded trained classes: {classes}")

# Re-define the same CNN architecture structure
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
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Initialize MTCNN Face Detector
print("[INFO] Initializing MTCNN Deep Learning Face Detector...")
detector = MTCNN(keep_all=False, device=device, min_face_size=40)

# 4. Image Preprocessing Transform Pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 5. Connect to IP Camera Stream
print(f"[INFO] Connecting to IP camera stream...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"[ERROR] Could not connect to RTSP stream at {RTSP_URL}")
    exit(1)

print("\n[INFO] Real-time Recognize, Cache & Track pipeline started.")
print(" - Press [q] to exit.\n")

# Caching Variables
cached_box = None
cached_label = ""
cached_color = (0, 0, 255)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from stream. Reconnecting...")
        continue

    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()

    # LOGIC: Run full Detector & Classifier ONLY when cache is empty or refresh interval is met
    if cached_box is None or (frame_count % CACHE_REFRESH_INTERVAL == 0):
        boxes, probs = detector.detect(rgb_frame)
        
        if boxes is not None and probs[0] > DETECTION_THRESHOLD:
            x1, y1, x2, y2 = boxes[0].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Update cached box ROI
                cached_box = [x1, y1, x2, y2]
                
                # Run Classifier Neural Network to Recognize Identity
                input_tensor = transform(face_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    prob_score, pred_idx = torch.max(probabilities, dim=1)
                    
                    score = prob_score.item()
                    
                    if score >= RECOGNITION_THRESHOLD:
                        predicted_name = classes[pred_idx.item()]
                        cached_color = (0, 255, 0) # Green
                        cached_label = f"Cached: {predicted_name.upper()} ({score * 100:.1f}%)"
                    else:
                        cached_color = (0, 0, 255) # Red
                        cached_label = f"Cached: UNKNOWN (?) [{score * 100:.1f}%]"
            else:
                cached_box = None
        else:
            cached_box = None

    # TRACKING PHASE: Reuse cached recognition data across frames without re-running heavy models
    if cached_box is not None:
        x1, y1, x2, y2 = cached_box
        
        # Draw bounding box and cached identity label
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), cached_color, 2)
        cv2.putText(display_frame, cached_label, (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cached_color, 2)
    else:
        # If no face is tracked/cached
        cv2.putText(display_frame, "Searching & Recognizing Face...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show live stream feed
    cv2.imshow("Recognize, Cache & Track Pipeline", display_frame)
    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Pipeline stopped.")