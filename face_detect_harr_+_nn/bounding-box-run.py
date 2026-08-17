import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget
from mss import mss

# ==================== CONFIGURATION ====================
MODEL_WEIGHTS_PATH = "face_cnn_weights.pth"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
IMG_SIZE = 128
RECOGNITION_THRESHOLD = 0.75  # Confidence cutoff, else show '?'
MARGIN = 10                   # Edge threshold in pixels for resizing overlay
# =======================================================


class ResizableOverlay(QWidget):
    def __init__(self, model, face_cascade, device, classes, transform):
        super().__init__()
        self.model = model
        self.face_cascade = face_cascade
        self.device = device
        self.classes = classes
        self.transform = transform

        self.pred_class = "Waiting..."
        self.conf_score = 0.0
        self.is_recognized = False

        # Window Setup: Frameless, Always on Top, Translucent background
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Initial geometry (Position X, Y, Width, Height)
        self.setGeometry(400, 200, 400, 400)

        # Interaction states
        self.drag_position = QPoint()
        self.resize_direction = None

        # Initialize MSS for screen capture inside window bounds
        self.sct = mss()

        # Timer to periodically capture screen region and run inference (~15 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_inference)
        self.timer.start(60)

        print("[INFO] Real-time Haar + PyTorch recognition overlay started.")
        print("Instructions:")
        print(" - Position the transparent bounding box over the face on your screen.")
        print(" - Drag the body to move, hover over edges to resize.")
        print(" - Press [Esc] to exit.\n")

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()

    def run_inference(self):
        try:
            geom = self.geometry()
            region = {
                "top": geom.y(),
                "left": geom.x(),
                "width": max(50, geom.width()),
                "height": max(50, geom.height())
            }

            sct_img = self.sct.grab(region)
            frame = np.array(sct_img)  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade inside the overlay area
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            face_found = False
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_found = True
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(width, x + w + pad_w)
                    y2 = min(height, y + h + pad_h)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            prob_score, pred_idx = torch.max(probabilities, dim=1)

                            score = prob_score.item()
                            if score >= RECOGNITION_THRESHOLD:
                                self.pred_class = self.classes[pred_idx.item()].upper()
                                self.conf_score = score * 100
                                self.is_recognized = True
                            else:
                                self.pred_class = "UNKNOWN (?)"
                                self.conf_score = score * 100
                                self.is_recognized = False
                    break

            if not face_found:
                self.pred_class = "No Face Detected!"
                self.conf_score = 0.0
                self.is_recognized = False

            self.update()
        except Exception:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent dark background inside the box
        painter.setBrush(QColor(0, 0, 0, 30))

        # Color-code border based on recognition state
        if self.is_recognized:
            pen_color = QColor(0, 255, 0)     # Green for recognized
        elif "No Face" in self.pred_class or "Waiting" in self.pred_class:
            pen_color = QColor(255, 255, 0)   # Yellow for searching/waiting
        else:
            pen_color = QColor(255, 0, 0)     # Red for unknown

        pen = QPen(pen_color, 3, Qt.SolidLine)
        painter.setPen(pen)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

        # Draw classification label text inside top-left corner
        painter.setPen(QPen(pen_color, 2))
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        if "No Face" in self.pred_class or "Waiting" in self.pred_class:
            text = self.pred_class
        else:
            text = f"{self.pred_class} ({self.conf_score:.1f}%)"
        
        painter.drawText(15, 25, text)

    def get_resize_direction(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()

        left = x < MARGIN
        right = x > w - MARGIN
        top = y < MARGIN
        bottom = y > h - MARGIN

        if left and top: return "top_left"
        if right and top: return "top_right"
        if left and bottom: return "bottom_left"
        if right and bottom: return "bottom_right"
        if left: return "left"
        if right: return "right"
        if top: return "top"
        if bottom: return "bottom"
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos()
            self.resize_direction = self.get_resize_direction(event.pos())
            self.setFocus()

    def mouseMoveEvent(self, event):
        global_pos = event.globalPos()

        if not event.buttons() & Qt.LeftButton:
            direction = self.get_resize_direction(event.pos())
            if direction in ["left", "right"]:
                self.setCursor(Qt.SizeHorCursor)
            elif direction in ["top", "bottom"]:
                self.setCursor(Qt.SizeVerCursor)
            elif direction in ["top_left", "bottom_right"]:
                self.setCursor(Qt.SizeFDiagCursor)
            elif direction in ["top_right", "bottom_left"]:
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        diff = global_pos - self.drag_position
        geom = self.geometry()

        if self.resize_direction:
            if "right" in self.resize_direction:
                geom.setWidth(max(150, geom.width() + diff.x()))
            if "bottom" in self.resize_direction:
                geom.setHeight(max(150, geom.height() + diff.y()))
            if "left" in self.resize_direction:
                new_width = max(150, geom.width() - diff.x())
                if new_width != 150:
                    geom.setLeft(geom.left() + diff.x())
            if "top" in self.resize_direction:
                new_height = max(150, geom.height() - diff.y())
                if new_height != 150:
                    geom.setTop(geom.top() + diff.y())
            self.setGeometry(geom)
        else:
            self.move(self.pos() + diff)

        self.drag_position = global_pos

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("[INFO] Exiting recognition overlay.")
            self.close()
            sys.exit(0)


def main():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Load Trained PyTorch Classifier Model & Classes
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[ERROR] Trained weights '{MODEL_WEIGHTS_PATH}' not found! Please run 'train.py' first.")
        return

    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    classes = checkpoint['classes']
    print(f"[INFO] Loaded trained classes: {classes}")

    class FaceClassifierCNN(nn.Module):
        def __init__(self, num_classes):
            super(FaceClassifierCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = FaceClassifierCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Load Haar Cascade Face Detector
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"[ERROR] Haar Cascade file '{HAAR_CASCADE_PATH}' not found!")
        return

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    print("[INFO] Haar Cascade loaded successfully.")

    # 4. Image Preprocessing Transform Pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. Start PyQt Application Overlay
    app = QApplication(sys.argv)
    overlay = ResizableOverlay(model, face_cascade, device, classes, transform)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()