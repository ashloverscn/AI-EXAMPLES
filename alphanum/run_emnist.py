import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- 1. Enhanced Residual Block & Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancedResCNN(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedResCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.res_block1 = ResidualBlock(32)
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.res_block2 = ResidualBlock(64)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.mid_conv(x)
        x = self.res_block2(x)
        
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        all_pts = np.concatenate(contours) if len(contours) > 1 else contours[0]
        x, y, w, h = cv2.boundingRect(all_pts)
        
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + (2 * margin))
        h = min(img.shape[0] - y, h + (2 * margin))
        
        img = img[y:y+h, x:x+w]
        
        if w > 5 and h > 5:
            aspect_ratio = w / h
            if w > h:
                new_w, new_h = 20, int(20 / aspect_ratio)
            else:
                new_h, new_w = 20, int(20 * aspect_ratio)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            final_img = np.zeros((28, 28), dtype=np.uint8)
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
            final_img[top:top+new_h, left:left+new_w] = img
            img = final_img
        else:
            return None
    else:
        return None

    # Normalization matching the training pipeline
    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    return torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0)


def main():
    dataset_path = "./dataset/train"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")
        return

    # Automatically read folder names sorted alphabetically to map indices to characters exactly like ImageFolder does
    emnist_classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(emnist_classes)
    print(f"Loaded {num_classes} character classes dynamically from folders.")

    device = torch.device("cpu")
    model = EnhancedResCNN(num_classes=num_classes).to(device)
    
    MODEL_PATH = "emnist_cnn.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights file '{MODEL_PATH}' not found! Please run your EMNIST training script first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    drawing = False

    def draw(event, x, y, flags, param):
        nonlocal drawing, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(canvas, (x, y), 20, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), 20, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    window_name = 'EMNIST Alphanumeric Drawing Board'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, draw)

    print("\n[INFO] Ready. Draw a character inside the window.")
    print("Controls:")
    print(" - Press 'd' after drawing to detect/predict.")
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
            cv2.setWindowTitle(window_name, "EMNIST Drawing Board - Cleared")
        elif key == ord('d'):
            input_tensor = preprocess_image(canvas.copy())
            if input_tensor is not None:
                with torch.no_grad():
                    output = model(input_tensor.to(device))
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    pred_idx = predicted.item()
                    conf_score = confidence.item()
                    pred_char = emnist_classes[pred_idx] if pred_idx < len(emnist_classes) else str(pred_idx)
                    
                    print(f"--> Detected Character: {pred_char}  (Confidence: {conf_score:.2f})")
                    cv2.setWindowTitle(window_name, f"Prediction: {pred_char} ({conf_score:.2f})")
            else:
                print("--> No drawing detected to predict!")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()