import os
import pickle
import sys
import tarfile
import time
import urllib.request
from PIL import Image

# Configuration
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
RAW_DATA_DIR = "./data"
DATASET_DIR = "./dataset"
FILENAME = "cifar-10-python.tar.gz"

start_time = None


def progress_bar(block_num, block_size, total_size):
  global start_time
  if block_num == 0:
    start_time = time.time()
    return

  downloaded = block_num * block_size
  duration = time.time() - start_time
  speed = downloaded / duration if duration > 0 else 0.0

  if total_size > 0:
    percent = min(float(downloaded) / total_size, 1.0)
    bar_length = 25
    filled_length = int(bar_length * percent)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    downloaded_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    speed_kb = speed / 1024

    sys.stdout.write(
        f"\r[{bar}] {percent * 100:.1f}% | "
        f"{downloaded_mb:.2f} / {total_mb:.2f} MB | "
        f"{speed_kb:.1f} KB/s"
    )
    sys.stdout.flush()
  else:
    downloaded_mb = downloaded / (1024 * 1024)
    sys.stdout.write(
        f"\rDownloaded {downloaded_mb:.2f} MB (unknown total size)"
    )
    sys.stdout.flush()


def unpickle(file):
  with open(file, "rb") as fo:
    dict_val = pickle.load(fo, encoding="bytes")
  return dict_val


def extract_images_to_dataset():
  extracted_path = os.path.join(RAW_DATA_DIR, "cifar-10-batches-py")

  print("\nLoading metadata...")
  meta_file = os.path.join(extracted_path, "batches.meta")
  meta = unpickle(meta_file)
  label_names = [label.decode("utf-8") for label in meta[b"label_names"]]

  # Process Training Batches
  print("Extracting training images into dataset/train/...")
  for i in range(1, 6):
    batch_file = os.path.join(extracted_path, f"data_batch_{i}")
    batch_data = unpickle(batch_file)
    data = batch_data[b"data"]
    labels = batch_data[b"labels"]

    for idx, (img_data, label_idx) in enumerate(zip(data, labels)):
      img_np = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
      img = Image.fromarray(img_np)

      class_name = label_names[label_idx]
      class_dir = os.path.join(DATASET_DIR, "train", class_name)
      os.makedirs(class_dir, exist_ok=True)
      img.save(os.path.join(class_dir, f"batch_{i}_img_{idx}.png"))

  # Process Test Batch
  print("Extracting test images into dataset/test/...")
  test_file = os.path.join(extracted_path, "test_batch")
  test_data = unpickle(test_file)
  data = test_data[b"data"]
  labels = test_data[b"labels"]

  for idx, (img_data, label_idx) in enumerate(zip(data, labels)):
    img_np = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img_np)

    class_name = label_names[label_idx]
    class_dir = os.path.join(DATASET_DIR, "test", class_name)
    os.makedirs(class_dir, exist_ok=True)
    img.save(os.path.join(class_dir, f"test_img_{idx}.png"))

  print(
      f"Images extracted successfully into {DATASET_DIR}/ (train and test"
      " folders)"
  )


def download_and_extract():
  os.makedirs(RAW_DATA_DIR, exist_ok=True)
  filepath = os.path.join(RAW_DATA_DIR, FILENAME)

  if not os.path.exists(filepath):
    print("Downloading CIFAR-10 dataset...")
    try:
      urllib.request.urlretrieve(URL, filepath, reporthook=progress_bar)
      print("\nDownload complete.")
    except Exception as e:
      print(f"\nError downloading dataset: {e}")
      return
  else:
    print(f"Archive already exists at {filepath}.")

  extracted_path = os.path.join(RAW_DATA_DIR, "cifar-10-batches-py")
  if not os.path.exists(extracted_path):
    print("Extracting raw binary archive files into ./data...")
    with tarfile.open(filepath, "r:gz") as tar:
      tar.extractall(path=RAW_DATA_DIR)
    print("Raw extraction complete.")
  else:
    print("Raw binary files already extracted inside ./data.")

  extract_images_to_dataset()


if __name__ == "__main__":
  download_and_extract()