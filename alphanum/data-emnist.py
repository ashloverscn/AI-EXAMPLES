import os
from torchvision import datasets
import torchvision.transforms.functional as TF

def extract_emnist_to_images():
    images_dir = "./dataset/train"
    
    # EMNIST Balanced split has 47 classes
    # The dataset provides mapping classes via train_dataset.classes or we can map them
    # EMNIST balanced classes mapping: 0-9 are digits, 10-35/etc are letters depending on the split.
    # We will let torchvision/dataset handle class mapping dynamically.
    
    print("Downloading/Loading raw EMNIST 'balanced' dataset using torchvision...")
    # download=True will fetch the compressed archive and extract it to ./data
    raw_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True)
    
    # Get the class names/mapping (EMNIST characters list)
    # EMNIST characters typically include ASCII mapping codes or strings
    # We can retrieve the character labels using raw_dataset.classes
    classes = raw_dataset.classes # A list representing the character or label names
    
    print(f"Total classes found in EMNIST Balanced split: {len(classes)}")

    # Check if images are already extracted and non-empty across all classes
    all_exist = True
    for class_name in classes:
        # Convert class name to string representation suitable for folder name
        folder_name = str(class_name)
        if isinstance(class_name, int):
            # If classes are integers, map them using chr() if they correspond to ASCII characters
            folder_name = chr(class_name) if class_name > 9 else str(class_name)
            
        class_dir = os.path.join(images_dir, folder_name)
        if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
            all_exist = False
            break

    if all_exist:
        print("[INFO] Extracted EMNIST image dataset already exists at './dataset/train/'. Skipping extraction.")
        return

    # Create directories for each class
    for class_idx, class_val in enumerate(classes):
        folder_name = chr(class_val) if isinstance(class_val, int) and class_val > 9 else str(class_val)
        os.makedirs(os.path.join(images_dir, folder_name), exist_ok=True)

    print("Extracting and saving individual EMNIST image files into category folders...")
    counts = {class_val: 0 for class_val in classes}

    for idx, (img, label_idx) in enumerate(raw_dataset):
        # EMNIST images are originally transposed/rotated by NIST. 
        # Fix orientation: Rotate 90 degrees clockwise and mirror horizontally.
        img = TF.rotate(img, -90)
        img = TF.hflip(img)

        label_val = classes[label_idx]
        folder_name = chr(label_val) if isinstance(label_val, int) and label_val > 9 else str(label_val)
        
        save_path = os.path.join(images_dir, folder_name, f"img_{counts[label_val]:05d}.png")
        img.save(save_path)
        counts[label_val] += 1

    print("\n[SUCCESS] All EMNIST samples successfully extracted to individual image files!")
    for class_val in classes:
        folder_name = chr(class_val) if isinstance(class_val, int) and class_val > 9 else str(class_val)
        print(f" - Class '{folder_name}': {counts[class_val]} images saved in ./dataset/train/{folder_name}/")

if __name__ == '__main__':
    extract_emnist_to_images()
