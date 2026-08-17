import os
import torch
import torchvision
import torchvision.transforms as transforms

# Import the correct model architecture class from train.py
from train import CIFARClassifierCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare test dataset from the image folder to grab correct class names
    test_dir = "./dataset/test"
    if not os.path.exists(test_dir):
        print(
            f"Error: Test directory not found at {test_dir}. Please run your"
            " downloader script first."
        )
        return

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ),
    ])

    testset = torchvision.datasets.ImageFolder(
        root=test_dir, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True
    )

    classes = testset.classes

    # 2. Load the trained weights from root directory
    model_path = "./cifar_cnn.pth"
    if not os.path.exists(model_path):
        print(
            f"Error: Model weights not found at {model_path}. Please run train.py"
            " first."
        )
        return

    model = CIFARClassifierCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")

    # 3. Grab a batch and run inference
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    print("\nRunning inference on a sample batch...")
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    print("-" * 45)
    print(f"{'Actual Label':<18} | {'Predicted Label':<18} | Result")
    print("-" * 45)
    for i in range(len(labels)):
        actual = classes[labels[i]]
        pred = classes[predicted[i]]
        status = "CORRECT" if actual == pred else "INCORRECT"
        print(f"{actual:<18} | {pred:<18} | {status}")
    print("-" * 45)


if __name__ == "__main__":
    main()