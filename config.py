import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
RAW_DATASET = os.path.join(BASE_DIR, "dataset", "raw")
SAMPLE_DATASET = os.path.join(BASE_DIR, "dataset", "sample")

# Processed Data
FRAME_OUTPUT = os.path.join(BASE_DIR, "dataset", "processed", "frames")
FACE_OUTPUT = os.path.join(BASE_DIR, "dataset", "processed", "faces")

# Image Settings
FRAME_INTERVAL = 10
IMAGE_SIZE = (224, 224)

MODEL_DIR = os.path.join(BASE_DIR, "models")

YUNET_MODEL = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx"
)