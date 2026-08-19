import os
import cv2

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
DATASET_DIR = "./dataset/train"
TARGET_IMAGES = 30         # Total images to collect per sign
IMG_SIZE = 64              # Standard size to save images (matches HOG pipeline)
# =======================================================

# 1. Get ASL Sign Name (e.g., 'a', 'b', 'hello')
sign_name = input("Enter ASL sign/letter name (e.g., a, b, c): ").strip().lower()
sign_dir = os.path.join(DATASET_DIR, sign_name)
os.makedirs(sign_dir, exist_ok=True)

# 2. Connect to IP Camera Stream
print(f"[INFO] Connecting to IP camera stream...")
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"[ERROR] Could not connect to RTSP stream at {RTSP_URL}")
    exit(1)

count = 0
print(f"\n[INFO] Collection started for ASL sign: '{sign_name.upper()}'")
print("Instructions:")
print(" - Position your hand inside the rectangle box on the screen.")
print(" - Press [SPACEBAR] to capture & save the hand crop.")
print(" - Press [q] to quit early.\n")

while count < TARGET_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from stream. Reconnecting...")
        continue

    height, width = frame.shape[:2]

    # Define a fixed capture box in the center of the frame (e.g., 250x250 pixels)
    box_w, box_h = 250, 250
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    display_frame = frame.copy()

    # Draw guiding rectangle box for hand placement
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    status_text = f"Sign: {sign_name.upper()} | Collected: {count}/{TARGET_IMAGES}"
    cv2.putText(display_frame, status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press SPACE to Capture | 'q' to Quit", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Extract hand crop from inside the box
    hand_crop = frame[y1:y2, x1:x2]

    # Show live feed window
    cv2.imshow("ASL Hand Dataset Collector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Press SPACE to save the cropped hand image
    if key == ord(' ') and hand_crop.size > 0:
        count += 1
        # Uniform resizing so training features match standard dimensions
        resized_hand = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
        
        save_path = os.path.join(sign_dir, f"{count:03d}.jpg")
        cv2.imwrite(save_path, resized_hand)
        print(f"[SAVED] {save_path}")

    # Press 'q' to quit early
    elif key == ord('q'):
        print("[INFO] Exiting collection manually.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Finished! Successfully saved {count} hand images to '{sign_dir}/'.")