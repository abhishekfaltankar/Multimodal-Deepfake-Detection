import os
import cv2
import mediapipe as mp

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import FRAME_OUTPUT, FACE_OUTPUT, IMAGE_SIZE

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)