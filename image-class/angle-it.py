import os
import cv2

# ==================== CONFIGURATION ====================
DATA_DIR = "./dataset/train"
IMG_SIZE = 64  # Resize back to uniform dimensions during augmentation
# =======================================================

def augment_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Dataset directory '{DATA_DIR}' not found!")
        return

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if len(classes) == 0:
        print(f"[ERROR] No class folders found inside '{DATA_DIR}'!")
        return

    print(f"[INFO] Starting image rotation augmentation for classes: {classes}")

    total_original = 0
    total_augmented = 0

    for cls_name in classes:
        cls_path = os.path.join(DATA_DIR, cls_name)
        image_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n[INFO] Processing class '{cls_name.upper()}' ({len(image_files)} original images)...")
        
        for img_name in image_files:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            total_original += 1
            base_name, ext = os.path.splitext(img_name)

            # Define the 4 rotation angles (0, 90, 180, 270)
            rotations = {
                0: cv2.ROTATE_90_CLOCKWISE,       # Placeholder, handled separately below
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }

            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated = img.copy()
                    suffix = "rot0"
                elif angle == 90:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    suffix = "rot90"
                elif angle == 180:
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                    suffix = "rot180"
                elif angle == 270:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    suffix = "rot270"

                # Ensure dimensions remain uniform
                rotated = cv2.resize(rotated, (IMG_SIZE, IMG_SIZE))

                # Save rotated duplicate back into the class folder
                new_filename = f"{base_name}_{suffix}{ext}"
                save_path = os.path.join(cls_path, new_filename)
                cv2.imwrite(save_path, rotated)
                total_augmented += 1

        print(f"[DONE] Augmented class '{cls_name.upper()}' successfully.")

    print(f"\n[SUMMARY] Augmentation complete!")
    print(f" - Original images processed: {total_original}")
    print(f" - Total images after rotation expansion: {total_augmented}")
    print("You can now re-run 'train.py' to update your 'knn_model_data.npz' model with the new angles.")

if __name__ == "__main__":
    augment_dataset()
