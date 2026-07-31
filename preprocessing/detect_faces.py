import os
import cv2

from config import FRAME_OUTPUT, YUNET_MODEL

# Load YuNet model
detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
    config="",
    input_size=(320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

# Test image
image_path = os.path.join(
    FRAME_OUTPUT,
    "real",
    "id0_0001",
    "frame_0000.jpg"
)

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not load image.")
    exit()

height, width = image.shape[:2]

# Tell YuNet the image size
detector.setInputSize((width, height))

# Detect faces
_, faces = detector.detect(image)

if faces is None:
    print("❌ No face detected.")
else:
    print(f"✅ {len(faces)} face(s) detected.")

    for face in faces:
        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

output_path = "test_result.jpg"
cv2.imwrite(output_path, image)

print(f"✅ Saved: {output_path}")