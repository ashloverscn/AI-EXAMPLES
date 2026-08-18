import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_name = "efficientdet_lite0.tflite"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
        urllib.request.urlretrieve(url, model_name)

    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened(): return

    base_options = python.BaseOptions(model_asset_path=model_name)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)

    with vision.ObjectDetector.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            h, w, _ = frame.shape
            for detection in result.detections:
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                
                # Draw bounding box and label
                cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
                category = detection.categories[0]
                text = f"{category.category_name} ({category.score:.2f})"
                cv2.putText(frame, text, (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Object Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
