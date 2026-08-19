import os
import cv2
import numpy as np

# ==================== CONFIGURATION ====================
DATA_DIR = "./dataset/train"
MODEL_SAVE_PATH = "knn_model_data.npz"
IMG_SIZE = 64
K = 3  # Number of neighbors for KNN classification
# =======================================================

# Initialize OpenCV HOG Descriptor for shape features
hog = cv2.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Ensure uniform image dimensions
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Compute HOG feature vector
    features = hog.compute(img)
    if features is not None:
        return features.flatten().astype(np.float32)
    return None

def load_dataset():
    X = []
    y = []
    
    # Discover class directories (a, b, c, etc.)
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset directory '{DATA_DIR}' not found!")
        return None, None, None

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    if len(classes) == 0:
        print(f"[ERROR] No class folders found inside '{DATA_DIR}'!")
        return None, None, None

    print(f"[INFO] Detected ASL Classes: {classes}")

    for idx, cls_name in enumerate(classes):
        cls_path = os.path.join(DATA_DIR, cls_name)
        count = 0
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            features = extract_hog_features(img_path)
            if features is not None:
                X.append(features)
                y.append(idx)
                count += 1
        print(f"[INFO] Loaded {count} samples for class '{cls_name.upper()}'")

    X_train = np.array(X, dtype=np.float32)
    y_train = np.array(y, dtype=np.int64)

    # Normalize feature vectors (crucial for accurate Euclidean distance calculation in KNN)
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-9)

    return X_train, y_train, np.array(classes)

print("[INFO] Processing dataset and extracting HOG features...")
X_train, y_train, classes = load_dataset()

if X_train is None or len(X_train) == 0:
    print("[ERROR] Training aborted due to empty dataset.")
    exit(1)

print(f"[INFO] Total training samples: {len(X_train)} | Feature vector size: {X_train.shape[1]}")

# Save training data, labels, and class names into the required npz file format
np.savez(
    MODEL_SAVE_PATH,
    X_train=X_train,
    y_train=y_train,
    classes=classes
)

print(f"[SUCCESS] Training features successfully saved to '{MODEL_SAVE_PATH}'!")
print("You can now run your PyQt5 bounding box overlay script to test real-time sign recognition.")