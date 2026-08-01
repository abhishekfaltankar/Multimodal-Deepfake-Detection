import os
import cv2

from config import FRAME_OUTPUT, FACE_OUTPUT, IMAGE_SIZE, YUNET_MODEL


def load_detector():
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL,
        config="",
        input_size=(320, 320),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


def detect_face(detector, image):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    return faces

def process_image(detector, image_path, output_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not read {image_path}")
        return

    faces = detect_face(detector, image)

    if faces is None:
        print(f"⚠️ No face found in {image_path}")
        return

    # Take the first detected face
    x, y, w, h = faces[0][:4].astype(int)

    face = image[y:y+h, x:x+w]

    face = cv2.resize(face, IMAGE_SIZE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, face)

    print(f"✅ Saved: {output_path}")
    
def process_video(detector, input_folder, output_folder):

    for image_name in os.listdir(input_folder):

        input_path = os.path.join(input_folder, image_name)

        output_path = os.path.join(output_folder, image_name)

        process_image(
            detector,
            input_path,
            output_path
        )
        
def process_dataset(detector):

    for label in ["real", "fake"]:

        input_root = os.path.join(FRAME_OUTPUT, label)

        output_root = os.path.join(FACE_OUTPUT, label)

        for video_folder in os.listdir(input_root):

            input_folder = os.path.join(
                input_root,
                video_folder
            )

            output_folder = os.path.join(
                output_root,
                video_folder
            )

            os.makedirs(output_folder, exist_ok=True)

            print(f"Processing {video_folder}...")

            process_video(
                detector,
                input_folder,
                output_folder
            )
    
if __name__ == "__main__":

    detector = load_detector()

    input_image = os.path.join(
        FRAME_OUTPUT,
        "real",
        "id0_0001",
        "frame_0000.jpg"
    )

    output_image = os.path.join(
        FACE_OUTPUT,
        "real",
        "test_face.jpg"
    )

    process_dataset(detector)