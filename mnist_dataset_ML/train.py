import os
import cv2
import numpy as np

# Try importing CuPy for GPU acceleration; fallback to NumPy if unavailable
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("--- GPU Accelerated Mode Enabled (CuPy) ---")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("--- CPU Mode Enabled (Install 'cupy' for GPU acceleration) ---")

DATA_DIR = "dataset/train"
IMG_SIZE = 32
CHECKPOINT_FILE = "knn_model_data.npz"

print("--- Loading Dataset from Directory ---")
data = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' not found.")
    exit()

class_names = sorted([cls for cls in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, cls))])
print(f"Detected Classes: {class_names}")

# Read images with standard loop
for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(DATA_DIR, class_name)
    img_files = os.listdir(class_folder)
    print(f"Loading class '{class_name}' ({len(img_files)} images)...")
    
    for img_file in img_files:
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img_resized.flatten())
            labels.append(idx)

X_all = np.array(data, dtype='float32') / 255.0
y_all = np.array(labels, dtype='int')

total_samples = X_all.shape[0]
if total_samples == 0:
    print("Error: No training images found!")
    exit()

# Shuffle dataset randomly
np.random.seed(42)
indices = np.arange(total_samples)
np.random.shuffle(indices)
X_all = X_all[indices]
y_all = y_all[indices]

# 80% Train, 20% Validation split for Overfitting Protection
split_idx = int(total_samples * 0.8)
X_train_np, X_val_np = X_all[:split_idx], X_all[split_idx:]
y_train_np, y_val_np = y_all[:split_idx], y_all[split_idx:]

print(f"\nTraining samples: {len(X_train_np)} | Validation samples: {len(X_val_np)}")

# --- Checkpoint Recovery Logic ---
best_k = 1
best_accuracy = -1
tested_k_history = []

if os.path.exists(CHECKPOINT_FILE):
    print(f"--- Found existing checkpoint '{CHECKPOINT_FILE}'. Loading previous learning data... ---")
    try:
        chk = np.load(CHECKPOINT_FILE, allow_pickle=True)
        saved_best_k = int(chk["best_k"])
        # If the checkpoint has recorded a valid evaluation, initialize from it
        if "best_accuracy" in chk:
            best_accuracy = float(chk["best_accuracy"])
            best_k = saved_best_k
            print(f"Resumed from checkpoint! Previous Best K: {best_k} | Best Accuracy: {best_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Warning: Could not fully read checkpoint file ({e}). Starting fresh training evaluation.")

print("--- Optimizing K (Overfitting Protection via Validation Set) ---")

# Move data to GPU if available
if GPU_AVAILABLE:
    X_train = cp.asarray(X_train_np)
    X_val = cp.asarray(X_val_np)
    y_train = cp.asarray(y_train_np)
    y_val = cp.asarray(y_val_np)
else:
    X_train, X_val = X_train_np, X_val_np
    y_train, y_val = y_train_np, y_val_np

k_values = [1, 3, 5, 7, 9, 11]

for k in k_values:
    if k > len(X_train_np):
        continue
    
    correct = 0
    for i in range(len(X_val)):
        # Compute Euclidean distance (accelerated via GPU if CuPy is active)
        distances = cp.linalg.norm(X_train - X_val[i], axis=1)
        k_indices = cp.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        
        # Majority vote (convert back to numpy briefly for bincount if on GPU)
        if GPU_AVAILABLE:
            k_nearest_labels_cpu = cp.asnumpy(k_nearest_labels)
        else:
            k_nearest_labels_cpu = k_nearest_labels
            
        counts = np.bincount(k_nearest_labels_cpu)
        predicted_label = np.argmax(counts)
        
        target_label = y_val_np[i] if GPU_AVAILABLE else y_val[i]
        if predicted_label == target_label:
            correct += 1
            
    accuracy = correct / len(X_val_np)
    print(f"Tested K={k} | Validation Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

    # --- Save Checkpoint After Every K Check ---
    np.savez(CHECKPOINT_FILE, 
             X_train=X_train_np, 
             y_train=y_train_np, 
             best_k=best_k, 
             best_accuracy=best_accuracy,
             class_names=np.array(class_names))
    print(f"[Checkpoint Saved] Progress recorded for K={k}.")

print(f"\n[Overfitting Protection Completed] Selected Optimal K: {best_k} (Validation Accuracy: {best_accuracy * 100:.2f}%)")
print(f"Final training state securely saved to '{CHECKPOINT_FILE}'.")