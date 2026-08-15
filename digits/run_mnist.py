import torch
import torch.nn as nn
import cv2
import numpy as np
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
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

    img_tensor = img.astype(np.float32) / 255.0
    return torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0)

def main():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    MODEL_PATH = "mnist_cnn.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights file '{MODEL_PATH}' not found!")
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

    window_name = 'MNIST Drawing Board'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, draw)

    print("\n[INFO] Ready. Draw a digit inside the window.")
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
            cv2.setWindowTitle(window_name, "MNIST Drawing Board - Cleared")
        elif key == ord('d'):
            input_tensor = preprocess_image(canvas.copy())
            if input_tensor is not None:
                with torch.no_grad():
                    output = model(input_tensor.to(device))
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    pred_digit = predicted.item()
                    conf_score = confidence.item()
                    
                    print(f"--> Detected Digit: {pred_digit}  (Confidence: {conf_score:.2f})")
                    cv2.setWindowTitle(window_name, f"Prediction: {pred_digit} ({conf_score:.2f})")
            else:
                print("--> No drawing detected to predict!")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()