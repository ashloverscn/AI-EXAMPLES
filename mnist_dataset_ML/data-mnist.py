import os
from torchvision import datasets

def extract_mnist_to_images():
    images_dir = "./dataset/train"
    
    # Check if images are already extracted and non-empty across all classes
    all_exist = True
    for i in range(10):
        class_dir = os.path.join(images_dir, str(i))
        if not os.path.exists(class_dir) or len(os.listdir(class_dir)) == 0:
            all_exist = False
            break

    if all_exist:
        print("[INFO] Extracted image dataset already exists at './dataset/train/'. Skipping extraction.")
        return

    # Create directories
    for i in range(10):
        os.makedirs(os.path.join(images_dir, str(i)), exist_ok=True)

    print("Downloading/Loading raw MNIST dataset using torchvision...")
    raw_dataset = datasets.MNIST(root='./data', train=True, download=True)
    
    print("Extracting and saving individual image files into category folders...")
    counts = {i: 0 for i in range(10)}

    for idx, (img, label) in enumerate(raw_dataset):
        save_path = os.path.join(images_dir, str(label), f"img_{counts[label]:05d}.png")
        img.save(save_path)
        counts[label] += 1

    print("\n[SUCCESS] All MNIST samples successfully extracted to individual image files!")
    for i in range(10):
        print(f" - Digit {i}: {counts[i]} images saved in ./dataset/train/{i}/")

if __name__ == '__main__':
    extract_mnist_to_images()
