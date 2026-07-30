import os
import cv2

# Input and Output Paths
INPUT_PATH = "dataset/sample"
OUTPUT_PATH = "dataset/processed/frames"

FRAME_INTERVAL = 10  # Save every 10th frame

# Process both folders
for label in ["real", "fake"]:

    input_folder = os.path.join(INPUT_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(input_folder):

        video_path = os.path.join(input_folder, video_file)

        cap = cv2.VideoCapture(video_path)

        video_name = os.path.splitext(video_file)[0]

        save_folder = os.path.join(output_folder, video_name)
        os.makedirs(save_folder, exist_ok=True)

        frame_count = 0
        saved_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:

                frame_path = os.path.join(
                    save_folder,
                    f"frame_{saved_count:04d}.jpg"
                )

                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()

        print(f"{video_file} -> {saved_count} frames extracted")

print("\n✅ Frame extraction completed!")