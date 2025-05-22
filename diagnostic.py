import cv2
import numpy as np
from ultralytics import YOLO

from utils import read_video, save_video


def diagnose_video(video_path, model_path, output_path):
    """
    Diagnose issues with object detection on a video.
    Creates a diagnostic video showing raw detections.
    """
    print(f"Reading video from {video_path}")
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")

    # Load the model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Process frames in smaller batches to avoid memory issues
    batch_size = 10
    output_frames = []

    for i in range(0, len(video_frames), batch_size):
        print(f"Processing frames {i} to {min(i+batch_size, len(video_frames))}")
        batch = video_frames[i : i + batch_size]

        # Run detection with lower confidence threshold
        results = model.predict(batch, conf=0.05)

        for j, result in enumerate(results):
            frame = batch[j].copy()

            # Draw all detections with class names and confidence
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = result.names[cls]

                # Different colors for different classes
                if cls_name == "player":
                    color = (0, 255, 0)  # Green
                elif cls_name == "ball":
                    color = (0, 0, 255)  # Red
                elif cls_name == "referee":
                    color = (255, 0, 0)  # Blue
                elif cls_name == "goalkeeper":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with confidence
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            # Add frame count
            cv2.putText(
                frame,
                f"Frame: {i+j}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Add detection count
            cv2.putText(
                frame,
                f"Detections: {len(boxes)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            output_frames.append(frame)

    # Save diagnostic video
    print(f"Saving diagnostic video to {output_path}")
    save_video(output_frames, output_path)
    print("Diagnostic complete")


if __name__ == "__main__":
    # Change these paths to match your setup
    video_path = "input_videos/demo_vid_2.mp4"  # Your new video
    model_path = "models/best.pt"
    output_path = "output_videos/diagnostic.avi"

    diagnose_video(video_path, model_path, output_path)
    diagnose_video(video_path, model_path, output_path)
