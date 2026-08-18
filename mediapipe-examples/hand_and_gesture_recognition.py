import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_name = "hand_landmarker.task"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_name)

    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error opening RTSP stream.")
        return

    base_options = python.BaseOptions(model_asset_path=model_name)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

    with vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks in result.hand_landmarks:
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            cv2.imshow("Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()