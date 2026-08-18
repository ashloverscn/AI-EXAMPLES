import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import text

def main():
    model_name = "language_detector.tflite"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        url = "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite"
        urllib.request.urlretrieve(url, model_name)
        print("Download complete!")

    # Configure the Language Detector
    base_options = python.BaseOptions(model_asset_path=model_name)
    options = text.LanguageDetectorOptions(base_options=base_options)

    print("Loading Language Detector model...")
    with text.LanguageDetector.create_from_options(options) as detector:
        
        # Test multilingual text snippets
        samples = [
            "Hello, how are you doing today?",
            "Bonjour, comment allez-vous aujourd'hui ?",
            "Hola, como estas hoy?"
        ]

        for sample in samples:
            print(f"\nText: '{sample}'")
            result = detector.detect(sample)
            
            for prediction in result.detections:
                print(f" -> Language Code: {prediction.language_code} | Probability: {prediction.probability:.4f}")

if __name__ == "__main__":
    main()