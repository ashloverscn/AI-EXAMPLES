import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
MODEL_WEIGHTS_PATH = "face_cnn_weights.pth"
IMG_SIZE = 128
DETECTION_THRESHOLD = 0.90   # Caffe face detection confidence
RECOGNITION_THRESHOLD = 0.75 # Classifier confidence to identify person, else '?'

# Caffe Model Paths
PROTOTXT_PATH = "deploy.prototxt"
CAFFE_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
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

# 3. Initialize OpenCV Caffe Deep Learning Face Detector
print("[INFO] Initializing OpenCV Caffe DNN Face Detector...")
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFE_MODEL_PATH):
    print(f"[ERROR] Caffe model files ('{PROTOTXT_PATH}' or '{CAFFE_MODEL_PATH}') not found!")
    exit(1)

detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)

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

print("\n[INFO] Real-time neural recognition started.")
print(" - Press [q] to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from stream. Reconnecting...")
        continue

    height, width = frame.shape[:2]
    
    # Prepare blob for OpenCV Caffe model (300x300, scale, mean subtraction)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 117.0, 123.0], False, False)
    detector.setInput(blob)
    detections = detector.forward()

    display_frame = frame.copy()
    face_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > DETECTION_THRESHOLD:
            face_detected = True
            
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            x1, y1, x2, y2 = box.astype(int)

            # Ensure coordinates stay within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                # Preprocess face for PyTorch CNN classifier
                input_tensor = transform(face_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    prob_score, pred_idx = torch.max(probabilities, dim=1)
                    
                    score = prob_score.item()
                    
                    # Check if recognition confidence is high enough
                    if score >= RECOGNITION_THRESHOLD:
                        predicted_name = classes[pred_idx.item()]
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{predicted_name.upper()} ({score * 100:.1f}%)"
                    else:
                        # Unknown face or low confidence -> Show '?'
                        color = (0, 0, 255)  # Red for unknown
                        label = f"UNKNOWN (?) [{score * 100:.1f}%]"

                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # If no face is detected at all in the frame
    if not face_detected:
        cv2.putText(display_frame, "No Face Present (?)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show live recognition feed
    cv2.imshow("Caffe SSD + PyTorch Face Recognition", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recognition stopped.")