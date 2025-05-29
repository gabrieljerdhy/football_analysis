import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
sys.path.append('../')


class FieldKeypointsDetector:
    """
    Detects field keypoints using YOLO model and provides goal area detection capabilities.
    """
    
    def __init__(self, model_path="models/best_fk.pt", confidence_threshold=0.7):
        """
        Initialize the field keypoints detector.
        
        Args:
            model_path (str): Path to the YOLO field keypoints model
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Standard football field keypoints mapping
        # These represent the typical keypoints that should be detected
        self.keypoint_classes = {
            0: "goal_left_post_top",
            1: "goal_left_post_bottom", 
            2: "goal_right_post_top",
            3: "goal_right_post_bottom",
            4: "penalty_area_left_top",
            5: "penalty_area_left_bottom",
            6: "penalty_area_right_top", 
            7: "penalty_area_right_bottom",
            8: "center_circle_top",
            9: "center_circle_bottom",
            10: "center_circle_left",
            11: "center_circle_right",
            12: "halfway_line_top",
            13: "halfway_line_bottom"
        }
        
        # Goal-related keypoints for each goal
        self.left_goal_keypoints = ["goal_left_post_top", "goal_left_post_bottom"]
        self.right_goal_keypoints = ["goal_right_post_top", "goal_right_post_bottom"]
        
        # Cache for detected keypoints
        self.detected_keypoints = {}
        self.goal_areas = {"left": None, "right": None}
        
    def detect_keypoints(self, frame):
        """
        Detect field keypoints in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: Dictionary of detected keypoints with their positions
        """
        results = self.model(frame, conf=self.confidence_threshold)
        
        detected_keypoints = {}
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if conf >= self.confidence_threshold:
                    class_id = int(class_id)
                    if class_id in self.keypoint_classes:
                        keypoint_name = self.keypoint_classes[class_id]
                        
                        # Get center point of the bounding box
                        x_center = int((box[0] + box[2]) / 2)
                        y_center = int((box[1] + box[3]) / 2)
                        
                        detected_keypoints[keypoint_name] = {
                            'position': (x_center, y_center),
                            'bbox': box,
                            'confidence': conf
                        }
        
        self.detected_keypoints = detected_keypoints
        self._update_goal_areas()
        return detected_keypoints
    
    def _update_goal_areas(self):
        """
        Update goal area boundaries based on detected keypoints.
        """
        # Left goal area
        left_goal_points = []
        for keypoint in self.left_goal_keypoints:
            if keypoint in self.detected_keypoints:
                left_goal_points.append(self.detected_keypoints[keypoint]['position'])
        
        if len(left_goal_points) >= 2:
            # Calculate goal area boundaries
            x_coords = [p[0] for p in left_goal_points]
            y_coords = [p[1] for p in left_goal_points]
            
            self.goal_areas["left"] = {
                'x_min': min(x_coords) - 50,  # Add some margin
                'x_max': max(x_coords) + 50,
                'y_min': min(y_coords),
                'y_max': max(y_coords)
            }
        
        # Right goal area  
        right_goal_points = []
        for keypoint in self.right_goal_keypoints:
            if keypoint in self.detected_keypoints:
                right_goal_points.append(self.detected_keypoints[keypoint]['position'])
                
        if len(right_goal_points) >= 2:
            x_coords = [p[0] for p in right_goal_points]
            y_coords = [p[1] for p in right_goal_points]
            
            self.goal_areas["right"] = {
                'x_min': min(x_coords) - 50,
                'x_max': max(x_coords) + 50, 
                'y_min': min(y_coords),
                'y_max': max(y_coords)
            }
    
    def get_goal_areas(self):
        """
        Get the current goal area boundaries.
        
        Returns:
            dict: Goal areas for left and right goals
        """
        return self.goal_areas
    
    def is_ball_in_goal_area(self, ball_position, goal_side="both"):
        """
        Check if ball is in a goal area.
        
        Args:
            ball_position (tuple): (x, y) position of the ball
            goal_side (str): "left", "right", or "both"
            
        Returns:
            str or None: Which goal the ball is in, or None if not in any goal
        """
        x, y = ball_position
        
        if goal_side in ["left", "both"] and self.goal_areas["left"]:
            left_area = self.goal_areas["left"]
            if (left_area['x_min'] <= x <= left_area['x_max'] and 
                left_area['y_min'] <= y <= left_area['y_max']):
                return "left"
        
        if goal_side in ["right", "both"] and self.goal_areas["right"]:
            right_area = self.goal_areas["right"]
            if (right_area['x_min'] <= x <= right_area['x_max'] and
                right_area['y_min'] <= y <= right_area['y_max']):
                return "right"
                
        return None
    
    def draw_keypoints(self, frame):
        """
        Draw detected keypoints on the frame.
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            frame: Frame with keypoints drawn
        """
        frame_copy = frame.copy()
        
        for keypoint_name, data in self.detected_keypoints.items():
            position = data['position']
            confidence = data['confidence']
            
            # Draw keypoint
            cv2.circle(frame_copy, position, 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"{keypoint_name}: {confidence:.2f}"
            cv2.putText(frame_copy, label, (position[0] + 10, position[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_copy
    
    def draw_goal_areas(self, frame):
        """
        Draw goal areas on the frame.
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            frame: Frame with goal areas drawn
        """
        frame_copy = frame.copy()
        
        # Draw left goal area
        if self.goal_areas["left"]:
            left_area = self.goal_areas["left"]
            cv2.rectangle(frame_copy, 
                         (left_area['x_min'], left_area['y_min']),
                         (left_area['x_max'], left_area['y_max']),
                         (255, 0, 0), 2)
            cv2.putText(frame_copy, "Left Goal", 
                       (left_area['x_min'], left_area['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw right goal area
        if self.goal_areas["right"]:
            right_area = self.goal_areas["right"]
            cv2.rectangle(frame_copy,
                         (right_area['x_min'], right_area['y_min']),
                         (right_area['x_max'], right_area['y_max']),
                         (0, 0, 255), 2)
            cv2.putText(frame_copy, "Right Goal",
                       (right_area['x_min'], right_area['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_copy
