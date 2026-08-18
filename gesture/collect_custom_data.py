import os
import cv2

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://192.168.29.251:5543/live/channel1"
DATASET_DIR = "dataset/train"
BOX_SIZE = 300
# =======================================================

def main():
    # Prompt user for the target class/letter they want to capture
    target_class = input("Enter the sign/letter you want to collect data for (e.g., A, B, C): ").strip().upper()
    if not target_class:
        print("[ERROR] Invalid class name.")
        return

    class_dir = os.path.join(DATASET_DIR, target_class)
    os.makedirs(class_dir, exist_ok=True)

    # Count existing images so we don't overwrite them
    existing_files = os.listdir(class_dir)
    img_counter = len(existing_files)
    print(f"[INFO] Saving images to '{class_dir}'. Current count: {img_counter}")

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"[ERROR] Could not open RTSP stream at {RTSP_URL}")
        return

    print("\n--- INSTRUCTIONS ---")
    print(f"1. Place your hand inside the square box for sign '{target_class}'.")
    print("2. Press SPACEBAR to capture an image.")
    print("3. Press 'q' to quit and finish collection.\n")

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

        # Draw the bounding box for user guidance
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x1 + BOX_SIZE, y1 + BOX_SIZE), (255, 0, 0), 2)
        
        info_text = f"Class: {target_class} | Captured: {img_counter} | Press SPACE to save"
        cv2.putText(display_frame, info_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Data Collection Tool", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            break
        
        # Press SPACEBAR to capture and save grayscale image
        elif key == ord(' '):
            if roi.size != 0:
                # Convert crop to grayscale (matching your model preprocessing format)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                img_name = os.path.join(class_dir, f"{target_class}_{img_counter:04d}.png")
                cv2.imwrite(img_name, gray_roi)
                print(f"[SAVED] {img_name}")
                img_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Data collection complete! Total images for '{target_class}': {img_counter}")

if __name__ == "__main__":
    main()
