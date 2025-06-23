#!/usr/bin/env python3
"""
SoccerNet to YOLO Format Converter

This script converts SoccerNet temporal action annotations into spatial bounding box 
annotations suitable for YOLOv8 training. It creates datasets for:
1. Enhanced ball detection with action context
2. Action-aware player detection
3. Multi-class action detection

Usage:
    python scripts/convert_soccernet_to_yolo.py --soccernet-dir data/soccernet
    python scripts/convert_soccernet_to_yolo.py --soccernet-dir data/soccernet --dataset ball-action-spotting
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class SoccerNetToYOLOConverter:
    """Converts SoccerNet temporal annotations to YOLO spatial annotations."""
    
    # Action spotting class mappings (17 classes)
    ACTION_SPOTTING_CLASSES = {
        "Ball out of play": 0,
        "Throw-in": 1, 
        "Foul": 2,
        "Indirect free-kick": 3,
        "Direct free-kick": 4,
        "Kick-off": 5,
        "Goal": 6,
        "Substitution": 7,
        "Offside": 8,
        "Shots on target": 9,
        "Shots off target": 10,
        "Clearance": 11,
        "Ball out of play": 12,
        "Throw-in": 13,
        "Corner": 14,
        "Yellow card": 15,
        "Red card": 16
    }
    
    # Ball action spotting class mappings (12 classes)
    BALL_ACTION_CLASSES = {
        "Pass": 0,
        "Drive": 1,
        "Header": 2,
        "High Pass": 3,
        "Out": 4,
        "Cross": 5,
        "Throw In": 6,
        "Shot": 7,
        "Ball Player Block": 8,
        "Player Successful Tackle": 9,
        "Free Kick": 10,
        "Goal": 11
    }
    
    def __init__(self, soccernet_dir: str, output_dir: str = "data/soccernet/yolo_datasets"):
        """
        Initialize the converter.
        
        Args:
            soccernet_dir: Path to SoccerNet data directory
            output_dir: Output directory for YOLO datasets
        """
        self.soccernet_dir = Path(soccernet_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        self.create_output_structure()
    
    def create_output_structure(self):
        """Create YOLO dataset directory structure."""
        datasets = [
            "enhanced_ball_detection",
            "action_aware_players", 
            "multi_class_actions"
        ]
        
        for dataset in datasets:
            for split in ["train", "val", "test"]:
                (self.output_dir / dataset / "images" / split).mkdir(parents=True, exist_ok=True)
                (self.output_dir / dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def load_annotations(self, annotation_file: Path) -> Dict:
        """Load SoccerNet annotations from JSON file."""
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def extract_frames_around_action(self, video_path: Path, timestamp: float, 
                                   context_seconds: float = 2.0) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames around an action timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Action timestamp in seconds
            context_seconds: Seconds of context around the action
            
        Returns:
            List of (frame, frame_timestamp) tuples
        """
        if not video_path.exists():
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range around action
        center_frame = int(timestamp * fps)
        context_frames = int(context_seconds * fps)
        start_frame = max(0, center_frame - context_frames)
        end_frame = min(total_frames, center_frame + context_frames)
        
        frames = []
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_timestamp = frame_idx / fps
                frames.append((frame, frame_timestamp))
        
        cap.release()
        return frames
    
    def create_synthetic_ball_bbox(self, frame: np.ndarray, action_type: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Create synthetic ball bounding box based on action type and frame analysis.
        
        This is a simplified approach - in practice, you'd want to use more sophisticated
        ball detection or tracking methods.
        
        Args:
            frame: Video frame
            action_type: Type of action (affects likely ball position)
            
        Returns:
            Bounding box (x1, y1, x2, y2) or None if no ball detected
        """
        height, width = frame.shape[:2]
        
        # Simple heuristic: assume ball is in center area for most actions
        # In practice, you'd use actual ball detection/tracking
        center_x, center_y = width // 2, height // 2
        
        # Ball size estimation (typical ball appears as 10-30 pixels)
        ball_size = 20
        
        # Adjust position based on action type
        if action_type in ["Shot", "Goal"]:
            # Shots often happen in penalty area
            center_y = int(height * 0.6)  # Lower part of field
        elif action_type in ["Corner", "Cross"]:
            # Corners happen at field edges
            center_x = int(width * 0.1) if np.random.random() > 0.5 else int(width * 0.9)
        elif action_type in ["Throw In"]:
            # Throw-ins happen at sidelines
            center_y = int(height * 0.3) if np.random.random() > 0.5 else int(height * 0.7)
        
        # Add some randomness
        center_x += np.random.randint(-50, 50)
        center_y += np.random.randint(-50, 50)
        
        # Ensure within frame bounds
        center_x = max(ball_size, min(width - ball_size, center_x))
        center_y = max(ball_size, min(height - ball_size, center_y))
        
        x1 = center_x - ball_size // 2
        y1 = center_y - ball_size // 2
        x2 = center_x + ball_size // 2
        y2 = center_y + ball_size // 2
        
        return (x1, y1, x2, y2)
    
    def bbox_to_yolo_format(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box to YOLO format (normalized center x, y, width, height)."""
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return (center_x, center_y, width, height)
    
    def convert_ball_action_spotting(self, split: str = "train"):
        """Convert ball action spotting data to YOLO format."""
        print(f"🔄 Converting ball action spotting data for {split} split...")
        
        # Find annotation files
        annotation_dir = self.soccernet_dir / "ball_action_spotting" / "annotations"
        video_dir = self.soccernet_dir / "ball_action_spotting" / "videos"
        
        output_images_dir = self.output_dir / "enhanced_ball_detection" / "images" / split
        output_labels_dir = self.output_dir / "enhanced_ball_detection" / "labels" / split
        
        annotation_files = list(annotation_dir.rglob("Labels-ball.json"))
        
        image_count = 0
        for annotation_file in tqdm(annotation_files, desc=f"Processing {split} annotations"):
            # Load annotations
            annotations = self.load_annotations(annotation_file)
            
            # Find corresponding video files
            game_path = annotation_file.parent
            video_files = list(game_path.parent.parent / "videos" / game_path.name).rglob("*.mkv")
            
            for video_file in video_files:
                if not video_file.exists():
                    continue
                
                # Process annotations for this video
                half = "1" if "1_" in video_file.name else "2"
                if half not in annotations["annotations"]:
                    continue
                
                for annotation in annotations["annotations"][half]:
                    action_type = annotation["label"]
                    timestamp = annotation["gameTime"]
                    
                    if action_type not in self.BALL_ACTION_CLASSES:
                        continue
                    
                    # Extract frames around action
                    frames = self.extract_frames_around_action(video_file, timestamp)
                    
                    for frame, frame_timestamp in frames:
                        # Create synthetic ball bounding box
                        ball_bbox = self.create_synthetic_ball_bbox(frame, action_type)
                        if ball_bbox is None:
                            continue
                        
                        # Save image
                        image_filename = f"{game_path.name}_{half}_{timestamp:.2f}_{frame_timestamp:.2f}.jpg"
                        image_path = output_images_dir / image_filename
                        cv2.imwrite(str(image_path), frame)
                        
                        # Create YOLO annotation
                        height, width = frame.shape[:2]
                        yolo_bbox = self.bbox_to_yolo_format(ball_bbox, width, height)
                        class_id = self.BALL_ACTION_CLASSES[action_type]
                        
                        # Save label
                        label_filename = image_filename.replace('.jpg', '.txt')
                        label_path = output_labels_dir / label_filename
                        with open(label_path, 'w') as f:
                            f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                        
                        image_count += 1
        
        print(f"✅ Created {image_count} training images for ball action spotting")
    
    def create_dataset_yaml(self, dataset_name: str, class_names: List[str]):
        """Create YOLO dataset configuration YAML file."""
        yaml_content = f"""# {dataset_name} dataset configuration for YOLOv8

# Dataset paths (relative to this file)
path: {self.output_dir / dataset_name}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
        
        yaml_path = self.output_dir / dataset_name / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"📄 Created dataset configuration: {yaml_path}")
    
    def convert_all_datasets(self):
        """Convert all SoccerNet datasets to YOLO format."""
        print("🚀 Starting SoccerNet to YOLO conversion...")
        
        # Convert ball action spotting
        if (self.soccernet_dir / "ball_action_spotting").exists():
            for split in ["train", "val", "test"]:
                self.convert_ball_action_spotting(split)
            
            # Create dataset YAML
            ball_class_names = list(self.BALL_ACTION_CLASSES.keys())
            self.create_dataset_yaml("enhanced_ball_detection", ball_class_names)
        
        # TODO: Add action spotting conversion
        # TODO: Add action-aware player detection conversion
        
        print("✅ SoccerNet to YOLO conversion complete!")


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(description="Convert SoccerNet annotations to YOLO format")
    parser.add_argument("--soccernet-dir", type=str, default="data/soccernet",
                       help="Path to SoccerNet data directory")
    parser.add_argument("--output-dir", type=str, default="data/soccernet/yolo_datasets",
                       help="Output directory for YOLO datasets")
    parser.add_argument("--dataset", choices=["ball-action-spotting", "action-spotting", "both"],
                       default="both", help="Which dataset to convert")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = SoccerNetToYOLOConverter(args.soccernet_dir, args.output_dir)
    
    # Convert datasets
    converter.convert_all_datasets()
    
    print(f"🎯 YOLO datasets created in: {args.output_dir}")


if __name__ == "__main__":
    main()
