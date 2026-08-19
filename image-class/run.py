import os
import cv2
import numpy as np

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
MODEL_DATA_PATH = "knn_model_data.npz"
IMG_SIZE = 64
K = 3                         # Number of neighbors for KNN classification
RECOGNITION_THRESHOLD = 50.0  # Minimum vote percentage to confirm a sign, else 'UNKNOWN'
# =======================================================

# 1. Load Trained NumPy KNN Model Data
if not os.path.exists(MODEL_DATA_PATH):
    print(f"[ERROR] Model file '{MODEL_DATA_PATH}' not found! Please run 'train.py' first.")
    exit(1)

data = np.load(MODEL_DATA_PATH, allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]
classes = data["classes"]
print(f"[INFO] Loaded classes: {list(classes)}")

# 2. Initialize HOG Descriptor (must match training parameters exactly)
hog = cv2.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

# 3. Connect to IP Camera RTSP Stream
print(f"[INFO] Connecting to IP camera stream...")
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"[ERROR] Could not connect to RTSP stream at {RTSP_URL}")
    exit(1)

print("[INFO] ASL IP Camera Classifier running. Press 'q' to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from stream. Reconnecting...")
        continue

    height, width = frame.shape[:2]

    # Define the same central ROI tracking box used during dataset collection (250x250)
    box_w, box_h = 250, 250
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Crop region of interest where the hand sign should be placed
    hand_crop = frame[y1:y2, x1:x2]

    # Preprocess crop for HOG extraction
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    features = hog.compute(resized)
    pred_class = "WAITING"
    confidence = 0.0
    is_recognized = False

    if features is not None:
        x_live = features.flatten().astype(np.float32)
        # Normalize live features exactly like training data
        x_live = x_live / (np.linalg.norm(x_live) + 1e-9)

        # --- Pure NumPy K-Nearest Neighbors (KNN) Inference ---
        distances = np.sqrt(np.sum((X_train - x_live) ** 2, axis=1))
        k_indices = np.argsort(distances)[:K]
        k_labels = y_train[k_indices]

        # Majority vote among K neighbors
        pred_idx = np.bincount(k_labels).argmax()
        vote_count = np.sum(k_labels == pred_idx)
        confidence = (vote_count / K) * 100

        if confidence >= RECOGNITION_THRESHOLD:
            pred_class = str(classes[pred_idx]).upper()
            is_recognized = True
        else:
            pred_class = "UNKNOWN"
            is_recognized = False

    # --- Draw UI Elements on Live Frame ---
    display_frame = frame.copy()
    
    # Color code box border: Green for recognized, Yellow for unknown/waiting
    box_color = (0, 255, 0) if is_recognized else (0, 255, 255)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

    # Label Text Overlay
    label_text = f"Sign: {pred_class} ({confidence:.0f}%)"
    cv2.putText(display_frame, label_text, (x1, y1 - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
    
    cv2.putText(display_frame, "Press 'q' to Quit", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show live feed window
    cv2.imshow("ASL IP Camera Classifier (Pure NumPy)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting classifier.")
        break

cap.release()
cv2.destroyAllWindows()