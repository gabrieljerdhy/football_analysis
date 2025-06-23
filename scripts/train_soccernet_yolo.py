#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for SoccerNet Datasets

This script trains YOLOv8 models on SoccerNet-derived datasets:
1. Enhanced ball detection with action context
2. Action-aware player detection  
3. Multi-class action detection

Usage:
    python scripts/train_soccernet_yolo.py --dataset enhanced_ball_detection
    python scripts/train_soccernet_yolo.py --dataset enhanced_ball_detection --epochs 100 --batch-size 16
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from ultralytics import YOLO

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class SoccerNetYOLOTrainer:
    """Trains YOLOv8 models on SoccerNet-derived datasets."""
    
    def __init__(self, dataset_dir: str = "data/soccernet/yolo_datasets"):
        """
        Initialize the trainer.
        
        Args:
            dataset_dir: Directory containing YOLO datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path("data/models/soccernet")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Available datasets
        self.available_datasets = {
            "enhanced_ball_detection": {
                "description": "Enhanced ball detection with action context",
                "base_model": "yolov8n.pt",  # Start with nano for ball detection
                "classes": 12
            },
            "action_aware_players": {
                "description": "Player detection with action awareness",
                "base_model": "yolov8s.pt",  # Small model for player detection
                "classes": 17
            },
            "multi_class_actions": {
                "description": "Multi-class action detection",
                "base_model": "yolov8m.pt",  # Medium model for complex actions
                "classes": 29  # Combined action classes
            }
        }
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """Validate that dataset exists and has required structure."""
        dataset_path = self.dataset_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_path}")
            return False
        
        # Check for required files
        yaml_file = dataset_path / "dataset.yaml"
        if not yaml_file.exists():
            print(f"❌ Dataset configuration not found: {yaml_file}")
            return False
        
        # Check for train/val directories
        for split in ["train", "val"]:
            images_dir = dataset_path / "images" / split
            labels_dir = dataset_path / "labels" / split
            
            if not images_dir.exists():
                print(f"❌ Images directory not found: {images_dir}")
                return False
            
            if not labels_dir.exists():
                print(f"❌ Labels directory not found: {labels_dir}")
                return False
            
            # Check if directories have content
            if not any(images_dir.iterdir()):
                print(f"⚠️ No images found in: {images_dir}")
                return False
        
        return True
    
    def load_dataset_config(self, dataset_name: str) -> Dict:
        """Load dataset configuration from YAML file."""
        yaml_file = self.dataset_dir / dataset_name / "dataset.yaml"
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def create_training_config(self, dataset_name: str, **kwargs) -> Dict:
        """Create training configuration."""
        dataset_info = self.available_datasets[dataset_name]
        
        config = {
            "data": str(self.dataset_dir / dataset_name / "dataset.yaml"),
            "model": dataset_info["base_model"],
            "epochs": kwargs.get("epochs", 100),
            "batch": kwargs.get("batch_size", 16),
            "imgsz": kwargs.get("image_size", 640),
            "device": kwargs.get("device", "auto"),
            "workers": kwargs.get("workers", 8),
            "project": str(self.models_dir),
            "name": f"{dataset_name}_training",
            "save_period": kwargs.get("save_period", 10),
            "patience": kwargs.get("patience", 50),
            "save": True,
            "plots": True,
            "val": True
        }
        
        # Add dataset-specific configurations
        if dataset_name == "enhanced_ball_detection":
            config.update({
                "lr0": 0.01,  # Higher learning rate for ball detection
                "weight_decay": 0.0005,
                "mosaic": 1.0,  # Data augmentation for small objects
                "mixup": 0.1,
                "copy_paste": 0.1
            })
        elif dataset_name == "action_aware_players":
            config.update({
                "lr0": 0.001,  # Lower learning rate for player detection
                "weight_decay": 0.0005,
                "mosaic": 0.8,
                "mixup": 0.0
            })
        elif dataset_name == "multi_class_actions":
            config.update({
                "lr0": 0.001,
                "weight_decay": 0.0005,
                "mosaic": 0.5,  # Less aggressive augmentation for complex scenes
                "mixup": 0.0
            })
        
        return config
    
    def train_model(self, dataset_name: str, **kwargs) -> str:
        """
        Train YOLOv8 model on specified dataset.
        
        Args:
            dataset_name: Name of the dataset to train on
            **kwargs: Training parameters
            
        Returns:
            Path to trained model
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not self.validate_dataset(dataset_name):
            raise ValueError(f"Dataset validation failed: {dataset_name}")
        
        print(f"🚀 Starting training for {dataset_name}")
        print(f"📝 Description: {self.available_datasets[dataset_name]['description']}")
        
        # Create training configuration
        config = self.create_training_config(dataset_name, **kwargs)
        
        # Load base model
        model = YOLO(config["model"])
        
        # Print training info
        print(f"🏗️ Base model: {config['model']}")
        print(f"📊 Dataset: {config['data']}")
        print(f"🔄 Epochs: {config['epochs']}")
        print(f"📦 Batch size: {config['batch']}")
        print(f"🖼️ Image size: {config['imgsz']}")
        print(f"💻 Device: {config['device']}")
        
        # Start training
        results = model.train(**config)
        
        # Get path to best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        
        print(f"✅ Training completed!")
        print(f"🏆 Best model saved to: {best_model_path}")
        
        return str(best_model_path)
    
    def evaluate_model(self, model_path: str, dataset_name: str) -> Dict:
        """Evaluate trained model on test set."""
        if not self.validate_dataset(dataset_name):
            raise ValueError(f"Dataset validation failed: {dataset_name}")
        
        print(f"📊 Evaluating model on {dataset_name} test set...")
        
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation on test set
        dataset_yaml = self.dataset_dir / dataset_name / "dataset.yaml"
        results = model.val(data=str(dataset_yaml), split="test")
        
        # Extract key metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "f1": results.box.f1
        }
        
        print(f"📈 Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def create_comparison_report(self, results: Dict[str, Dict]) -> str:
        """Create a comparison report of different models."""
        report_path = self.models_dir / "training_comparison.md"
        
        with open(report_path, 'w') as f:
            f.write("# SoccerNet YOLOv8 Training Results\n\n")
            f.write("## Model Comparison\n\n")
            f.write("| Dataset | mAP50 | mAP50-95 | Precision | Recall | F1 |\n")
            f.write("|---------|-------|----------|-----------|--------|----|\\n")
            
            for dataset_name, metrics in results.items():
                f.write(f"| {dataset_name} | {metrics['mAP50']:.4f} | {metrics['mAP50-95']:.4f} | "
                       f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n")
            
            f.write("\n## Training Details\n\n")
            for dataset_name in results.keys():
                dataset_info = self.available_datasets[dataset_name]
                f.write(f"### {dataset_name}\n")
                f.write(f"- **Description**: {dataset_info['description']}\n")
                f.write(f"- **Base Model**: {dataset_info['base_model']}\n")
                f.write(f"- **Classes**: {dataset_info['classes']}\n\n")
        
        return str(report_path)


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 models on SoccerNet datasets")
    parser.add_argument("--dataset", required=True,
                       choices=["enhanced_ball_detection", "action_aware_players", "multi_class_actions", "all"],
                       help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--dataset-dir", type=str, default="data/soccernet/yolo_datasets",
                       help="Directory containing YOLO datasets")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SoccerNetYOLOTrainer(args.dataset_dir)
    
    # Training parameters
    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "device": args.device,
        "workers": args.workers
    }
    
    results = {}
    
    # Train models
    if args.dataset == "all":
        datasets_to_train = ["enhanced_ball_detection", "action_aware_players", "multi_class_actions"]
    else:
        datasets_to_train = [args.dataset]
    
    for dataset_name in datasets_to_train:
        try:
            # Train model
            model_path = trainer.train_model(dataset_name, **train_params)
            
            # Evaluate if requested
            if args.evaluate:
                metrics = trainer.evaluate_model(model_path, dataset_name)
                results[dataset_name] = metrics
            
        except Exception as e:
            print(f"❌ Training failed for {dataset_name}: {e}")
            continue
    
    # Create comparison report if multiple models were trained
    if len(results) > 1:
        report_path = trainer.create_comparison_report(results)
        print(f"📄 Comparison report created: {report_path}")
    
    print("🎯 Training pipeline complete!")


if __name__ == "__main__":
    main()
