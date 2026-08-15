import cv2
import torch
import torch.nn as nn
import numpy as np
import os

# --- EMNIST Balanced Character Mapping (47 Classes) ---
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C / c', 'D', 'E', 'F', 'G', 'H', 'I / i', 'J / j', 
    'K / k', 'L / l', 'M / m', 'N', 'O / o', 'P / p', 'Q', 'R', 
    'S / s', 'T', 'U / u', 'V / v', 'W / w', 'X / x', 'Y / y', 'Z / z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

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
    def __init__(self, num_classes=47):
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
        self.fc2 = nn.Linear(128, num_classes)  # 47 classes for EMNIST Balanced
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

def enhance_and_preprocess_roi(roi_bgr):
    """
    Applies image enhancement filters to make the character prominent,
    handles uneven lighting, and returns both the preprocessed tensor
    and the filtered preview image for visualization.
    """
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr

    # 1. Bilateral Filter: Removes noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances contrast of faint characters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # 3. Adaptive Thresholding: Binarizes the image robustly against shadows/lighting
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # 4. Morphological operations to clean up small speckles and thicken strokes
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        all_pts = np.concatenate(contours) if len(contours) > 1 else contours[0]
        x, y, w, h = cv2.boundingRect(all_pts)
        
        # Filter out tiny noise contours
        if w < 12 or h < 12:
            return None, thresh

        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(thresh.shape[1] - x, w + (2 * margin))
        h = min(thresh.shape[0] - y, h + (2 * margin))
        
        digit_crop = thresh[y:y+h, x:x+w]
        
        if w > 5 and h > 5:
            aspect_ratio = w / h
            if w > h:
                new_w, new_h = 20, int(20 / aspect_ratio)
            else:
                new_h, new_w = 20, int(20 * aspect_ratio)
            
            digit_crop = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            final_img = np.zeros((28, 28), dtype=np.uint8)
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
            final_img[top:top+new_h, left:left+new_w] = digit_crop
            
            # Normalize to match EMNIST training spec & EnhancedResCNN normalization
            img_tensor = final_img.astype(np.float32) / 255.0
            img_tensor = (img_tensor - 0.1307) / 0.3081
            return torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0), thresh
            
    return None, thresh

def main():
    device = torch.device("cpu")
    num_classes = len(EMNIST_CLASSES)
    model = EnhancedResCNN(num_classes=num_classes).to(device)
    
    MODEL_PATH = "emnist_cnn.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights file '{MODEL_PATH}' not found! Please train the model first.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- Connect to IP Camera RTSP Stream ---
    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"
    print(f"Connecting to IP Camera stream: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("[Error] Could not open RTSP stream. Check connection and URL.")
        return

    window_name = 'IP Camera Filtered EMNIST Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n[INFO] Stream connected successfully!")
    print("Controls:")
    print(" - Place your character inside the green box.")
    print(" - Press 'q' to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame from RTSP stream.")
            break

        display_frame = cv2.resize(frame, (640, 480))
        h_frame, w_frame, _ = display_frame.shape

        # Define Static Green Box ROI dimensions (200x200 in the center)
        box_size = 200
        box_x1 = (w_frame - box_size) // 2
        box_y1 = (h_frame - box_size) // 2
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size

        # Extract ROI and pass through enhancement filters
        roi = display_frame[box_y1:box_y2, box_x1:box_x2]
        input_tensor, filtered_thresh = enhance_and_preprocess_roi(roi)
        
        # Show the real-time thresholded/filtered binary view in the top-left corner
        filtered_colored = cv2.cvtColor(filtered_thresh, cv2.COLOR_GRAY2BGR)
        filtered_resized = cv2.resize(filtered_colored, (100, 100))
        display_frame[10:110, 10:110] = filtered_resized
        cv2.putText(display_frame, "Filtered ROI", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        detected_text = "Waiting for character..."
        box_color = (0, 255, 0) # Green

        if input_tensor is not None:
            with torch.no_grad():
                output = model(input_tensor.to(device))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                pred_idx = predicted.item()
                conf_score = confidence.item()
                
                if conf_score > 0.60:
                    pred_char = EMNIST_CLASSES[pred_idx] if pred_idx < len(EMNIST_CLASSES) else str(pred_idx)
                    detected_text = f"Char: {pred_char} ({conf_score*100:.1f}%)"
                    box_color = (255, 0, 0) # Blue highlight on valid detection

        # Draw ROI box and decoded text overlay
        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
        cv2.putText(display_frame, detected_text, (box_x1, box_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()