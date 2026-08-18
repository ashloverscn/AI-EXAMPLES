import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model_if_needed():
    model_name = "pose_landmarker_full.task"
    if not os.path.exists(model_name):
        print(f"Downloading MediaPipe pose landmarker model ({model_name})...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_name)
        print("Download complete!")
    return model_name

def main():
    # Specify the RTSP camera stream URL
    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"

    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream. Please check the RTSP URL or network connection.")
        return

    # Ensure model file is present
    model_path = download_model_if_needed()

    # Initialize MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Loading Pose Landmarker model...")
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("Press 'q' or 'ESC' to exit the video window.")

        # Define manual pose connections since mp.solutions is unavailable in modern mediapipe
        POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (27, 31), (28, 32), (27, 31), (28, 32)
        ]

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to grab frame from RTSP stream.")
                break

            # Convert OpenCV BGR frame to MediaPipe Image format (RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform pose estimation
            detection_result = landmarker.detect(mp_image)

            # Draw pose landmarks if detected
            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    h, w, _ = frame.shape
                    
                    # Draw points
                    for landmark in landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                    # Draw skeleton lines
                    for connection in POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Display the resulting frame in an OpenCV window
            cv2.imshow("RTSP Pose Estimation", frame)

            # Exit cleanly if 'q' or 'ESC' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()