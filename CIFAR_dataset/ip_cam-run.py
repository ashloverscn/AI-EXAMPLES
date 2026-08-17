import os
import cv2
import torch
import torchvision.transforms as transforms

# Import the model class matching train.py
from train import CIFARClassifierCNN as SimpleCNN


def main():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2. CIFAR-10 Class Labels
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # 3. Load the Trained Model Weights
    model_path = "./cifar_cnn.pth"
    if not os.path.exists(model_path):
        print(
            f"[ERROR] Model weights not found at '{model_path}'. Please run"
            " 'train.py' first."
        )
        return

    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[INFO] Model weights loaded successfully.")

    # 4. Image Preprocessing Pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # 5. Connect to RTSP Stream
    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"
    print(f"[INFO] Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(
            "[ERROR] Could not open RTSP stream. Check connection or URL address."
        )
        return

    print(
        "[INFO] Live stream started. Press 'q' in the video window to exit."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from stream.")
            break

        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        roi = frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_roi).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            pred_class = classes[predicted_idx.item()]
            conf_score = confidence.item() * 100

        cv2.rectangle(
            frame,
            (start_x, start_y),
            (start_x + min_dim, start_y + min_dim),
            (0, 255, 0),
            2,
        )
        label_text = f"{pred_class} ({conf_score:.1f}%)"
        cv2.putText(
            frame,
            label_text,
            (start_x, max(30, start_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("CIFAR-10 RTSP Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")


if __name__ == "__main__":
    main()