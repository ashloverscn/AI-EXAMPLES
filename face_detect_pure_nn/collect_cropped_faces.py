import os
import cv2

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
DATASET_DIR = "dataset"
TARGET_IMAGES = 15         # Total cropped face images to collect
CONFIDENCE_THRESHOLD = 0.55  # Minimum detection confidence
# =======================================================

# 1. Load Caffe Face Detector using generic reader
print("[INFO] Loading Caffe face detector model...")
model_weights = "res10_300x300_ssd_iter_140000.caffemodel"
model_config = "deploy.prototxt"

if not os.path.exists(model_weights) or not os.path.exists(model_config):
    print("[ERROR] Model files ('deploy.prototxt' or '.caffemodel') not found in directory!")
    exit(1)

net = cv2.dnn.readNet(model_config, model_weights)

# 2. Get User Name
person_name = input("Enter person's name (e.g., anku or ashish): ").strip().lower()
person_dir = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_dir, exist_ok=True)

# 3. Connect to IP Camera Stream
print(f"[INFO] Connecting to IP camera stream...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"[ERROR] Could not connect to RTSP stream at {RTSP_URL}")
    exit(1)

count = 0
print(f"\n[INFO] Collection started for '{person_name}'.")
print("Instructions:")
print(" - Position your face in front of the camera.")
print(" - Press [SPACEBAR] to capture & save a cropped face.")
print(" - Press [q] to quit early.\n")

while count < TARGET_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from stream. Reconnecting...")
        continue

    height, width = frame.shape[:2]

    # Prepare frame for Caffe SSD detector
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104.0, 117.0, 123.0])
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0
    best_box = None

    # Find the most confident face in the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            if confidence > best_conf:
                best_conf = confidence
                box = detections[0, 0, i, 3:7] * [width, height, width, height]
                best_box = box.astype(int)

    display_frame = frame.copy()

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # Add a 15% padding margin around the detected face
        pad_w = int((x2 - x1) * 0.15)
        pad_h = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)

        # Draw green bounding box on live preview
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        status_text = f"Collected: {count}/{TARGET_IMAGES} | Press SPACE to Save"
        cv2.putText(display_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        face_crop = frame[y1:y2, x1:x2]
    else:
        cv2.putText(display_frame, "No Face Detected!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        face_crop = None

    # Show live feed window
    cv2.imshow("IP Camera Face Collector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Press SPACE to save the cropped face
    if key == ord(' ') and face_crop is not None and face_crop.size > 0:
        count += 1
        # Uniform resizing so every training sample matches (128x128)
        resized_face = cv2.resize(face_crop, (128, 128))
        
        save_path = os.path.join(person_dir, f"{count:03d}.jpg")
        cv2.imwrite(save_path, resized_face)
        print(f"[SAVED] {save_path}")

    # Press 'q' to quit early
    elif key == ord('q'):
        print("[INFO] Exiting collection manually.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Finished! Successfully saved {count} cropped faces to '{person_dir}/'.")