import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():
    model_name = "selfie_segmenter.tflite"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        urllib.request.urlretrieve(url, model_name)
        print("Download complete!")

    rtsp_url = "rtsp://192.168.29.251:5543/live/channel1"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    base_options = python.BaseOptions(model_asset_path=model_name)
    options = vision.ImageSegmenterOptions(
        base_options=base_options, 
        output_category_mask=True
    )

    print("Loading Image Segmenter model...")
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        print("Press 'q' to exit.")
        while cap.isOpened():
            success, frame = cap.read()
            if not success: 
                print("Warning: Failed to grab frame from stream.")
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Perform segmentation
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()

            # Squeeze to remove trailing single dimensions and create a proper boolean mask
            mask_2d = np.squeeze(category_mask)
            condition = np.stack((mask_2d,) * 3, axis=-1) > 0.1

            # Create background tint effect (solid blue background)
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = (255, 0, 0) # Blue background tint
            
            output_frame = np.where(condition, frame, bg_image)

            cv2.imshow("Image Segmentation Mask", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()