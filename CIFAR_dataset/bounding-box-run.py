import json
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget
from mss import mss

MODEL_PATH = "./cifar_cnn.pth"
CLASSES_JSON_PATH = "./classes.json"
MARGIN = 10  # Edge threshold in pixels for resizing


# Define the exact same architecture matching train.py
class CIFARClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFARClassifierCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResizableOverlay(QWidget):
    def __init__(self, model, device, classes, transform):
        super().__init__()
        self.model = model
        self.device = device
        self.classes = classes
        self.transform = transform

        self.pred_class = "Waiting..."
        self.conf_score = 0.0

        # Window Setup: Frameless, Always on Top, Translucent background
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)  # Required to track mouse for resizing cursors

        # Initial geometry (Position X, Y, Width, Height)
        self.setGeometry(400, 200, 300, 300)

        # Interaction states
        self.drag_position = QPoint()
        self.resize_direction = None

        # Initialize MSS for screen capture inside window bounds
        self.sct = mss()

        # Timer to periodically capture screen region and run inference (~10 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_inference)
        self.timer.start(100)

        print("[INFO] Resizable see-through overlay started.")
        print("[INFO] Hover over edges to resize, drag the body to move. Press 'Esc' to exit.")

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                self.pred_class = self.classes[predicted_idx.item()]
                self.conf_score = confidence.item() * 100

            self.update()
        except Exception:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent dark background inside the box
        painter.setBrush(QColor(0, 0, 0, 40))

        # Neon Green Bounding Box Pen
        pen = QPen(QColor(0, 255, 0), 3, Qt.SolidLine)
        painter.setPen(pen)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

        # Draw classification label text
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
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

    def mouseMoveEvent(self, event):
        global_pos = event.globalPos()

        # Update cursor shape based on position
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

        # Handle Resizing or Moving
        diff = global_pos - self.drag_position
        geom = self.geometry()

        if self.resize_direction:
            if "right" in self.resize_direction:
                geom.setWidth(max(100, geom.width() + diff.x()))
            if "bottom" in self.resize_direction:
                geom.setHeight(max(100, geom.height() + diff.y()))
            if "left" in self.resize_direction:
                new_width = max(100, geom.width() - diff.x())
                if new_width != 100:
                    geom.setLeft(geom.left() + diff.x())
            if "top" in self.resize_direction:
                new_height = max(100, geom.height() - diff.y())
                if new_height != 100:
                    geom.setTop(geom.top() + diff.y())
            self.setGeometry(geom)
        else:
            # Move window if dragging inside
            self.move(self.pos() + diff)

        self.drag_position = global_pos

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            sys.exit(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Class Labels from classes.json
    if not os.path.exists(CLASSES_JSON_PATH):
        print(
            f"[ERROR] Classes file not found at '{CLASSES_JSON_PATH}'. Please run 'train.py' first."
        )
        return

    with open(CLASSES_JSON_PATH, "r") as f:
        classes = json.load(f)
    print(f"[INFO] Loaded {len(classes)} classes from {CLASSES_JSON_PATH}: {classes}")

    # 2. Check Model Weights
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found at '{MODEL_PATH}'. Run 'train.py' first.")
        return

    model = CIFARClassifierCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Model weights loaded successfully.")

    # 3. Image Preprocessing Pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    app = QApplication(sys.argv)
    overlay = ResizableOverlay(model, device, classes, transform)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()