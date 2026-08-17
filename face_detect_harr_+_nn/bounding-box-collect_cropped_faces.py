import os
import cv2
import numpy as np
import sys
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget
from mss import mss

# ==================== CONFIGURATION ====================
DATASET_DIR = "dataset"
TARGET_IMAGES = 15           # Total cropped face images to collect
CONFIDENCE_THRESHOLD = 0.55  # Minimum detection confidence
MARGIN = 10                  # Edge threshold in pixels for resizing overlay
# =======================================================


class FaceCollectorOverlay(QWidget):
    def __init__(self, net, person_dir):
        super().__init__()
        self.net = net
        self.person_dir = person_dir

        self.count = 0
        self.best_box = None

        # Window Setup: Frameless, Always on Top, Translucent background
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Ensures key presses work reliably

        # Initial geometry (Position X, Y, Width, Height)
        self.setGeometry(400, 200, 400, 400)

        # Interaction states
        self.drag_position = QPoint()
        self.resize_direction = None

        # Initialize MSS for screen capture inside window bounds
        self.sct = mss()

        # Timer to capture screen region and process face detection (~15 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(60)

        print(f"\n[INFO] Collection started for '{os.path.basename(person_dir)}'.")
        print("Instructions:")
        print(" - Position the transparent bounding box over your face on the screen.")
        print(" - Drag the body to move, hover over edges to resize.")
        print(" - Press [SPACEBAR] while focused on the overlay to capture & save.")
        print(" - Press [Esc] to quit early.\n")

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()

    def process_frame(self):
        try:
            geom = self.geometry()
            region = {
                "top": geom.y(),
                "left": geom.x(),
                "width": max(50, geom.width()),
                "height": max(50, geom.height())
            }

            # Grab screen region inside the overlay box for detection preview
            sct_img = self.sct.grab(region)
            frame = np.array(sct_img)  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Convert to BGR

            height, width = frame.shape[:2]

            # Prepare frame for Caffe SSD detector
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104.0, 117.0, 123.0])
            self.net.setInput(blob)
            detections = self.net.forward()

            best_conf = 0
            self.best_box = None

            # Find the most confident face in the frame
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    if confidence > best_conf:
                        best_conf = confidence
                        box = detections[0, 0, i, 3:7] * [width, height, width, height]
                        self.best_box = box.astype(int)

            if self.best_box is not None:
                x1, y1, x2, y2 = self.best_box

                # Add a 15% padding margin around the detected face
                pad_w = int((x2 - x1) * 0.15)
                pad_h = int((y2 - y1) * 0.15)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(width, x2 + pad_w)
                y2 = min(height, y2 + pad_h)

                self.best_box = (x1, y1, x2, y2)

            self.update()
        except Exception as e:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent dark background inside the box
        painter.setBrush(QColor(0, 0, 0, 30))

        # Neon Green Bounding Box Pen
        pen = QPen(QColor(0, 255, 0), 3, Qt.SolidLine)
        painter.setPen(pen)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

        # Draw inner face bounding box if detected
        if self.best_box is not None:
            bx1, by1, bx2, by2 = self.best_box
            painter.setPen(QPen(QColor(0, 255, 0, 180), 2, Qt.DashLine))
            painter.drawRect(bx1, by1, bx2 - bx1, by2 - by1)

            # Show status text inside the UI window (NOT saved in image)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setFont(QFont("Arial", 11, QFont.Bold))
            status_text = f"Collected: {self.count}/{TARGET_IMAGES} | SPACE: Save"
            painter.drawText(15, 25, status_text)
        else:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setFont(QFont("Arial", 11, QFont.Bold))
            painter.drawText(15, 25, "No Face Detected!")

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
        # Press SPACE to save the completely clean screen region (without overlay UI)
        if event.key() == Qt.Key_Space:
            if self.best_box is not None:
                geom = self.geometry()
                
                # Temporarily hide the overlay window so mss captures only the clean screen underneath
                self.hide()
                QApplication.processEvents() # Ensure the window is fully hidden before capture
                
                region = {
                    "top": geom.y(),
                    "left": geom.x(),
                    "width": max(50, geom.width()),
                    "height": max(50, geom.height())
                }
                sct_img = self.sct.grab(region)
                clean_frame = np.array(sct_img)
                clean_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGRA2BGR)
                
                # Show the overlay window again immediately
                self.show()
                self.activateWindow()
                self.setFocus()

                x1, y1, x2, y2 = self.best_box
                face_crop = clean_frame[y1:y2, x1:x2]

                if face_crop is not None and face_crop.size > 0:
                    self.count += 1
                    resized_face = cv2.resize(face_crop, (128, 128))
                    
                    save_path = os.path.join(self.person_dir, f"{self.count:03d}.jpg")
                    cv2.imwrite(save_path, resized_face)
                    print(f"[SAVED] {save_path} (Progress: {self.count}/{TARGET_IMAGES})")

                    if self.count >= TARGET_IMAGES:
                        print(f"\n[INFO] Target reached! Successfully saved {self.count} cropped faces to '{self.person_dir}/'.")
                        self.close()
                        sys.exit(0)
                else:
                    print("[WARNING] Failed to crop face cleanly.")
            else:
                print("[WARNING] Cannot save: No face currently detected in the overlay!")

        # Press 'Esc' to quit early
        elif event.key() == Qt.Key_Escape:
            print("[INFO] Exiting collection manually.")
            self.close()
            sys.exit(0)


def main():
    # 1. Load Caffe Face Detector
    print("[INFO] Loading Caffe face detector model...")
    model_weights = "res10_300x300_ssd_iter_140000.caffemodel"
    model_config = "deploy.prototxt"

    if not os.path.exists(model_weights) or not os.path.exists(model_config):
        print(f"[ERROR] Model files ('{model_config}' or '{model_weights}') not found in directory!")
        return

    net = cv2.dnn.readNet(model_config, model_weights)

    # 2. Get User Name
    person_name = input("Enter person's name (e.g., anku or ashish): ").strip().lower()
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # 3. Start PyQt Application Overlay
    app = QApplication(sys.argv)
    overlay = FaceCollectorOverlay(net, person_dir)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()