#!/usr/bin/env python3
"""
Unified Football Model Training Pipeline

This module handles training of the unified multi-task YOLO model that combines:
1. Player/referee detection
2. Enhanced ball detection
3. Field keypoint detection

Training Strategy:
- Multi-task learning with weighted loss functions
- Progressive training stages
- Data augmentation optimized for football scenarios
- Transfer learning from existing models
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class UnifiedModelTrainer:
    """
    Trainer for unified football detection model.
    
    Features:
    - Multi-task loss function with adaptive weighting
    - Progressive training strategy
    - Data fusion from multiple sources
    - Performance monitoring and validation
    """
    
    def __init__(
        self,
        base_model: str = "yolov8m.pt",
        output_dir: str = "data/models/unified",
        device: Optional[str] = None
    ):
        """
        Initialize unified model trainer.
        
        Args:
            base_model: Base YOLO model for transfer learning
            output_dir: Directory for saving trained models
            device: Training device
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training configuration
        self.training_config = self._get_default_config()
        
        # Task weights for multi-task learning
        self.task_weights = {
            'players': 1.0,      # Standard weight for player detection
            'ball': 2.0,         # Higher weight for critical ball detection
            'keypoints': 1.5     # Medium weight for field structure
        }
        
        # Performance tracking
        self.training_metrics = {
            'players': {'mAP': [], 'loss': []},
            'ball': {'mAP': [], 'loss': []},
            'keypoints': {'mAP': [], 'loss': []},
            'total': {'loss': []}
        }
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'warmup_epochs': 3,
            'patience': 50,
            'save_period': 10,
            'val_split': 0.2,
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            }
        }
    
    def prepare_unified_dataset(
        self,
        player_data_path: str,
        ball_data_path: str,
        keypoint_data_path: str,
        output_path: str = "data/datasets/unified_football"
    ) -> str:
        """
        Prepare unified dataset from separate task datasets.
        
        Args:
            player_data_path: Path to player detection dataset
            ball_data_path: Path to ball detection dataset
            keypoint_data_path: Path to keypoint detection dataset
            output_path: Output path for unified dataset
            
        Returns:
            Path to unified dataset configuration
        """
        print("🔄 Preparing unified dataset...")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset structure
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Merge datasets
        unified_images, unified_labels = self._merge_datasets(
            player_data_path, ball_data_path, keypoint_data_path
        )
        
        # Split train/val
        train_split = int(len(unified_images) * (1 - self.training_config['val_split']))
        
        train_images = unified_images[:train_split]
        val_images = unified_images[train_split:]
        train_labels = unified_labels[:train_split]
        val_labels = unified_labels[train_split:]
        
        # Copy files and create labels
        self._copy_dataset_files(train_images, train_labels, output_dir / "images" / "train", output_dir / "labels" / "train")
        self._copy_dataset_files(val_images, val_labels, output_dir / "images" / "val", output_dir / "labels" / "val")
        
        # Create dataset configuration
        dataset_config = self._create_dataset_config(output_dir)
        
        print(f"✅ Unified dataset prepared: {dataset_config}")
        return dataset_config
    
    def _merge_datasets(
        self, 
        player_path: str, 
        ball_path: str, 
        keypoint_path: str
    ) -> Tuple[List[str], List[Dict]]:
        """Merge multiple datasets into unified format."""
        unified_images = []
        unified_labels = []
        
        # Load existing datasets
        player_data = self._load_dataset(player_path)
        ball_data = self._load_dataset(ball_path)
        keypoint_data = self._load_dataset(keypoint_path)
        
        # Find common images across datasets
        common_images = set(player_data.keys()) & set(ball_data.keys()) & set(keypoint_data.keys())
        
        print(f"Found {len(common_images)} common images across all datasets")
        
        for image_path in common_images:
            unified_images.append(image_path)
            
            # Merge labels with task-specific class offsets
            merged_labels = []
            
            # Player labels (classes 0-2: player, referee, goalkeeper)
            if image_path in player_data:
                for label in player_data[image_path]:
                    merged_labels.append({
                        'class': label['class'],  # Keep original class IDs
                        'bbox': label['bbox'],
                        'task': 'players'
                    })
            
            # Ball labels (class 3: ball)
            if image_path in ball_data:
                for label in ball_data[image_path]:
                    merged_labels.append({
                        'class': 3,  # Ball class ID
                        'bbox': label['bbox'],
                        'task': 'ball'
                    })
            
            # Keypoint labels (classes 4-18: field keypoints)
            if image_path in keypoint_data:
                for label in keypoint_data[image_path]:
                    merged_labels.append({
                        'class': label['class'] + 4,  # Offset keypoint classes
                        'bbox': label['bbox'],
                        'task': 'keypoints'
                    })
            
            unified_labels.append(merged_labels)
        
        return unified_images, unified_labels
    
    def _load_dataset(self, dataset_path: str) -> Dict:
        """Load dataset from YOLO format."""
        dataset = {}
        
        # Load from YOLO dataset structure
        images_dir = Path(dataset_path) / "images"
        labels_dir = Path(dataset_path) / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️ Dataset not found: {dataset_path}")
            return dataset
        
        # Process all image files
        for img_file in images_dir.rglob("*.jpg"):
            label_file = labels_dir / img_file.relative_to(images_dir).with_suffix(".txt")
            
            if label_file.exists():
                labels = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            labels.append({
                                'class': int(parts[0]),
                                'bbox': [float(x) for x in parts[1:5]]
                            })
                
                dataset[str(img_file)] = labels
        
        return dataset
    
    def _copy_dataset_files(self, images: List[str], labels: List[Dict], img_dir: Path, label_dir: Path):
        """Copy dataset files to unified structure."""
        import shutil
        
        for i, (img_path, img_labels) in enumerate(zip(images, labels)):
            # Copy image
            img_src = Path(img_path)
            img_dst = img_dir / f"{i:06d}.jpg"
            shutil.copy2(img_src, img_dst)
            
            # Create unified label file
            label_dst = label_dir / f"{i:06d}.txt"
            with open(label_dst, 'w') as f:
                for label in img_labels:
                    bbox = label['bbox']
                    f.write(f"{label['class']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    
    def _create_dataset_config(self, dataset_dir: Path) -> str:
        """Create dataset configuration file."""
        config = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 19,  # Total number of classes
            'names': [
                'player',           # 0
                'referee',          # 1
                'goalkeeper',       # 2
                'ball',             # 3
                'goal_left_top',    # 4
                'goal_left_bottom', # 5
                'goal_right_top',   # 6
                'goal_right_bottom', # 7
                'penalty_left_top', # 8
                'penalty_left_bottom', # 9
                'penalty_right_top', # 10
                'penalty_right_bottom', # 11
                'center_circle_top', # 12
                'center_circle_bottom', # 13
                'center_circle_left', # 14
                'center_circle_right', # 15
                'halfway_line_top', # 16
                'halfway_line_bottom', # 17
                'corner_flag'       # 18
            ]
        }
        
        config_path = dataset_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def train_unified_model(
        self,
        dataset_config: str,
        model_name: str = "unified_football_detector",
        resume: bool = False
    ) -> str:
        """
        Train unified football detection model.
        
        Args:
            dataset_config: Path to dataset configuration
            model_name: Name for the trained model
            resume: Whether to resume training
            
        Returns:
            Path to trained model
        """
        print("🚀 Starting unified model training...")
        
        # Initialize model
        model = YOLO(self.base_model)
        
        # Configure training parameters
        train_args = {
            'data': dataset_config,
            'epochs': self.training_config['epochs'],
            'batch': self.training_config['batch_size'],
            'lr0': self.training_config['learning_rate'],
            'weight_decay': self.training_config['weight_decay'],
            'momentum': self.training_config['momentum'],
            'warmup_epochs': self.training_config['warmup_epochs'],
            'patience': self.training_config['patience'],
            'save_period': self.training_config['save_period'],
            'device': self.device,
            'project': str(self.output_dir),
            'name': model_name,
            'resume': resume,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'source': None,
            'show': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'vid_stride': 1,
            'stream_buffer': False,
            'line_width': None,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'lr0': self.training_config['learning_rate'],
            'lrf': 0.01,
            'momentum': self.training_config['momentum'],
            'weight_decay': self.training_config['weight_decay'],
            'warmup_epochs': self.training_config['warmup_epochs'],
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': self.training_config['augmentation']['hsv_h'],
            'hsv_s': self.training_config['augmentation']['hsv_s'],
            'hsv_v': self.training_config['augmentation']['hsv_v'],
            'degrees': self.training_config['augmentation']['degrees'],
            'translate': self.training_config['augmentation']['translate'],
            'scale': self.training_config['augmentation']['scale'],
            'shear': self.training_config['augmentation']['shear'],
            'perspective': self.training_config['augmentation']['perspective'],
            'flipud': self.training_config['augmentation']['flipud'],
            'fliplr': self.training_config['augmentation']['fliplr'],
            'mosaic': self.training_config['augmentation']['mosaic'],
            'mixup': self.training_config['augmentation']['mixup'],
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0,
        }
        
        # Start training
        results = model.train(**train_args)
        
        # Save final model
        model_path = self.output_dir / model_name / "weights" / "best.pt"
        final_model_path = self.output_dir / f"{model_name}.pt"
        
        if model_path.exists():
            import shutil
            shutil.copy2(model_path, final_model_path)
            print(f"✅ Unified model saved: {final_model_path}")
        
        return str(final_model_path)
    
    def validate_model(self, model_path: str, dataset_config: str) -> Dict:
        """Validate trained unified model."""
        print("🔍 Validating unified model...")
        
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=dataset_config,
            device=self.device,
            batch=1,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=False,
            plots=True,
            save_json=True
        )
        
        # Extract metrics
        validation_metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16)
        }
        
        print(f"📊 Validation Results:")
        print(f"   mAP@0.5: {validation_metrics['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {validation_metrics['mAP50-95']:.3f}")
        print(f"   Precision: {validation_metrics['precision']:.3f}")
        print(f"   Recall: {validation_metrics['recall']:.3f}")
        print(f"   F1-Score: {validation_metrics['f1']:.3f}")
        
        return validation_metrics
