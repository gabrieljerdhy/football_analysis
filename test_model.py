import cv2
import numpy as np
from ultralytics import YOLO

from utils import read_video


def test_model_on_frames(video_path, model_path, num_frames=5):
    """
    Test the YOLO model on a few frames from the video
    and print detailed detection information.
    """
    print(f"Reading video from {video_path}")
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")

    if len(video_frames) == 0:
        print("ERROR: No frames were loaded from the video!")
        return

    # Load the model
    print(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    # Test on a few frames
    test_frames = [
        video_frames[i]
        for i in range(0, len(video_frames), len(video_frames) // num_frames)
    ]

    print(f"\nTesting model on {len(test_frames)} frames...")
    for i, frame in enumerate(test_frames):
        print(f"\nFrame {i} shape: {frame.shape}")

        try:
            # Try different confidence thresholds
            for conf in [0.5, 0.3, 0.1, 0.05]:
                results = model.predict(frame, conf=conf)
                boxes = results[0].boxes

                print(f"  Confidence threshold {conf}: {len(boxes)} detections")

                # Print details of each detection
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_val = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = results[0].names[cls]

                    print(
                        f"    Detection {j}: {cls_name}, conf={conf_val:.2f}, bbox=({x1},{y1},{x2},{y2})"
                    )

                # Save a sample image with detections for the lowest threshold
                if conf == 0.05:
                    output_frame = frame.copy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_val = float(box.conf[0])
                        cls = int(box.cls[0])
                        cls_name = results[0].names[cls]

                        # Different colors for different classes
                        if cls_name == "player":
                            color = (0, 255, 0)  # Green
                        elif cls_name == "ball":
                            color = (0, 0, 255)  # Red
                        else:
                            color = (255, 255, 255)  # White

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label with confidence
                        label = f"{cls_name}: {conf_val:.2f}"
                        cv2.putText(
                            output_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    cv2.imwrite(f"output_videos/test_frame_{i}.jpg", output_frame)
                    print(
                        f"  Saved detection image to output_videos/test_frame_{i}.jpg"
                    )

        except Exception as e:
            print(f"  ERROR processing frame: {e}")


if __name__ == "__main__":
    # Change these paths to match your setup
    video_path = "input_videos/demo_vid_2.mp4"  # Your new video
    model_path = "models/best.pt"

    test_model_on_frames(video_path, model_path)
    test_model_on_frames(video_path, model_path)
