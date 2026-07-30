import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset Paths
RAW_DATASET = os.path.join(BASE_DIR, "dataset", "raw")
SAMPLE_DATASET = os.path.join(BASE_DIR, "dataset", "sample")

FRAME_OUTPUT = os.path.join(BASE_DIR, "dataset", "processed", "frames")
FACE_OUTPUT = os.path.join(BASE_DIR, "dataset", "processed", "faces")

# Image Settings
FRAME_INTERVAL = 10
IMAGE_SIZE = (224, 224)