import os
from torchvision import datasets
import torchvision.transforms.functional as TF

# Standard EMNIST Balanced split mapping for 47 classes
EMNIST_BALANCED_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def extract_emnist_to_images():
    images_dir = "./dataset/train"
    
    print("Downloading/Loading raw EMNIST 'balanced' dataset using torchvision...")
    raw_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True)
    
    print(f"Total dataset items to process: {len(raw_dataset)}")

    # Check if all 47 category folders exist and are non-empty
    all_exist = True
    for char_label in EMNIST_BALANCED_CLASSES:
        class_dir = os.path.join(images_dir, char_label)
        if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
            all_exist = False
            break

    if all_exist:
        print("[INFO] Extracted EMNIST balanced dataset already exists at './dataset/train/' with all 47 classes populated. Skipping extraction.")
        return

    # Create directories for each of the 47 classes
    for char_label in EMNIST_BALANCED_CLASSES:
        os.makedirs(os.path.join(images_dir, char_label), exist_ok=True)

    print("Extracting and saving individual EMNIST image files into 47 category folders...")
    counts = {char_label: 0 for char_label in EMNIST_BALANCED_CLASSES}

    for idx, (img, label_idx) in enumerate(raw_dataset):
        # Fix orientation: Rotate 90 degrees clockwise and mirror horizontally
        img = TF.rotate(img, -90)
        img = TF.hflip(img)

        char_label = EMNIST_BALANCED_CLASSES[label_idx]
        
        save_path = os.path.join(images_dir, char_label, f"img_{counts[char_label]:05d}.png")
        img.save(save_path)
        counts[char_label] += 1

        if (idx + 1) % 20000 == 0:
            print(f"Processed {idx + 1} / {len(raw_dataset)} images...")

    # --- Verification Step ---
    print("\n--- Verifying Extraction Completeness ---")
    verification_passed = True
    total_files_extracted = 0

    for char_label in EMNIST_BALANCED_CLASSES:
        class_dir = os.path.join(images_dir, char_label)
        files = os.listdir(class_dir)
        num_files = len(files)
        total_files_extracted += num_files
        
        if num_files == 0:
            print(f"  [X] ERROR: Class '{char_label}' folder is empty!")
            verification_passed = False
        else:
            print(f"  [✓] Class '{char_label}': {num_files} images verified.")

    print(f"\nTotal files counted across all folders: {total_files_extracted} / {len(raw_dataset)}")

    if verification_passed and total_files_extracted == len(raw_dataset):
        print("\n[SUCCESS] All EMNIST Balanced elements have been properly downloaded, oriented, and verified!")
    else:
        print("\n[WARNING] Extraction completed, but file counts do not match completely. Please re-run.")

if __name__ == '__main__':
    extract_emnist_to_images()
