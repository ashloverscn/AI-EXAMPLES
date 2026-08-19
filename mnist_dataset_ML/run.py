import cv2
import numpy as np

# 1. Load Saved Traditional ML Model Data
print("Loading model data...")
try:
    data = np.load("knn_model_data.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    best_k = int(data["best_k"])
    class_names = data["class_names"]
    print(f"Model loaded successfully. Using K={best_k} | Classes: {list(class_names)}")
except FileNotFoundError:
    print("Error: 'knn_model_data.npz' not found. Please run your 'train.py' script first.")
    exit()

IMG_SIZE = 32
canvas_size = 400
canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(canvas, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

window_name = 'Traditional ML Drawing Board'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, draw)

print("\n[INFO] Ready. Draw your class inside the window.")
print("Controls:")
print(" - Press 'd' after drawing to classify/predict.")
print(" - Press 'c' to clear canvas.")
print(" - Press 'q' to quit.\n")

while True:
    cv2.imshow(window_name, canvas)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        print("--- Canvas Cleared ---")
        cv2.setWindowTitle(window_name, "Drawing Board - Cleared")
    elif key == ord('d'):
        # --- Preprocessing Logic Adapted from Drawing Board ---
        img = canvas.copy()
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            all_pts = np.concatenate(contours) if len(contours) > 1 else contours[0]
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            
            margin = 15
            bx = max(0, bx - margin)
            by = max(0, by - margin)
            bw = min(img.shape[1] - bx, bw + (2 * margin))
            bh = min(img.shape[0] - by, bh + (2 * margin))
            
            cropped = img[by:by+bh, bx:bx+bw]
            
            if bw > 5 and bh > 5:
                aspect_ratio = bw / bh
                if bw > bh:
                    new_w, new_h = IMG_SIZE - 4, int((IMG_SIZE - 4) / aspect_ratio)
                else:
                    new_h, new_w = IMG_SIZE - 4, int((IMG_SIZE - 4) * aspect_ratio)
                
                resized_roi = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Center inside target dimension matrix (IMG_SIZE x IMG_SIZE)
                final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                top = (IMG_SIZE - new_h) // 2
                left = (IMG_SIZE - new_w) // 2
                final_img[top:top+new_h, left:left+new_w] = resized_roi
                processed_face = final_img
            else:
                processed_face = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        else:
            processed_face = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Flatten and normalize
        flattened = processed_face.flatten().astype('float32') / 255.0

        # --- Traditional ML Prediction (k-NN Distance Calculation) ---
        distances = np.linalg.norm(X_train - flattened, axis=1)
        k_indices = np.argsort(distances)[:best_k]
        k_nearest_labels = y_train[k_indices]
        
        # Majority voting
        counts = np.bincount(k_nearest_labels, minlength=len(class_names))
        predicted_index = np.argmax(counts)
        predicted_label = class_names[predicted_index]
        
        # Confidence based on neighbor agreement ratio
        confidence = (counts[predicted_index] / best_k) * 100

        print(f"--> Predicted: {predicted_label} (Confidence: {confidence:.1f}%)")
        cv2.setWindowTitle(window_name, f"Prediction: {predicted_label} ({confidence:.1f}%)")

cv2.destroyAllWindows()