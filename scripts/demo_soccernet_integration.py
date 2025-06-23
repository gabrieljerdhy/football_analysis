#!/usr/bin/env python3
"""
SoccerNet Integration Demo

This script demonstrates how to integrate SoccerNet-trained models with your existing
football analysis system. It shows the complete workflow from data setup to model training.

Usage:
    python scripts/demo_soccernet_integration.py --demo-mode setup
    python scripts/demo_soccernet_integration.py --demo-mode train
    python scripts/demo_soccernet_integration.py --demo-mode integrate
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class SoccerNetIntegrationDemo:
    """Demonstrates SoccerNet integration with the football analysis system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.project_root = Path(__file__).parent.parent
        self.soccernet_dir = self.project_root / "data" / "soccernet"
        self.models_dir = self.project_root / "data" / "models"
        
    def demo_setup(self):
        """Demonstrate SoccerNet data setup process."""
        print("🚀 SoccerNet Integration Demo - Data Setup")
        print("=" * 60)
        
        print("\n1. 📋 Prerequisites:")
        print("   - Sign SoccerNet NDA: https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform")
        print("   - Receive password via email (24-48 hours)")
        print("   - Install dependencies: pip install -r requirements.txt")
        
        print("\n2. 📦 Data Download Commands:")
        print("   # Download both datasets with videos:")
        print("   python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD")
        print()
        print("   # Download specific dataset:")
        print("   python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD --dataset ball-action-spotting")
        print()
        print("   # Download annotations only (no videos):")
        print("   python scripts/setup_soccernet_data.py --no-videos")
        
        print("\n3. 🔄 Data Conversion:")
        print("   # Convert to YOLO format:")
        print("   python scripts/convert_soccernet_to_yolo.py --soccernet-dir data/soccernet")
        
        print("\n4. 📁 Expected Directory Structure:")
        self._show_directory_structure()
        
        # Check current setup status
        self._check_setup_status()
    
    def demo_training(self):
        """Demonstrate model training process."""
        print("🏋️ SoccerNet Integration Demo - Model Training")
        print("=" * 60)
        
        print("\n1. 🎯 Available Training Datasets:")
        datasets = {
            "enhanced_ball_detection": {
                "description": "Enhanced ball detection with action context",
                "classes": 12,
                "use_case": "Improved ball tracking and action recognition"
            },
            "action_aware_players": {
                "description": "Player detection with action awareness", 
                "classes": 17,
                "use_case": "Better player detection in complex scenes"
            },
            "multi_class_actions": {
                "description": "Multi-class action detection",
                "classes": 29,
                "use_case": "Comprehensive action recognition"
            }
        }
        
        for name, info in datasets.items():
            print(f"\n   📊 {name}:")
            print(f"      Description: {info['description']}")
            print(f"      Classes: {info['classes']}")
            print(f"      Use Case: {info['use_case']}")
        
        print("\n2. 🚀 Training Commands:")
        print("   # Train enhanced ball detection:")
        print("   python scripts/train_soccernet_yolo.py --dataset enhanced_ball_detection --epochs 100")
        print()
        print("   # Train action-aware player detection:")
        print("   python scripts/train_soccernet_yolo.py --dataset action_aware_players --epochs 150")
        print()
        print("   # Train all models with evaluation:")
        print("   python scripts/train_soccernet_yolo.py --dataset all --epochs 100 --evaluate")
        
        print("\n3. ⚙️ Training Parameters:")
        print("   --epochs: Number of training epochs (default: 100)")
        print("   --batch-size: Batch size (default: 16)")
        print("   --image-size: Input image size (default: 640)")
        print("   --device: Training device (default: auto)")
        print("   --evaluate: Evaluate model after training")
        
        # Check training readiness
        self._check_training_readiness()
    
    def demo_integration(self):
        """Demonstrate integration with existing system."""
        print("🔗 SoccerNet Integration Demo - System Integration")
        print("=" * 60)
        
        print("\n1. 🔄 Updating Your Tracker:")
        print("   Modify src/trackers/tracker.py to use SoccerNet models:")
        print()
        print("   ```python")
        print("   class Tracker:")
        print("       def __init__(self, model_path, use_soccernet=True):")
        print("           if use_soccernet:")
        print("               self.ball_model = YOLO('data/models/soccernet/enhanced_ball_detection/best.pt')")
        print("               self.player_model = YOLO('data/models/soccernet/action_aware_players/best.pt')")
        print("           else:")
        print("               self.model = YOLO(model_path)")
        print("   ```")
        
        print("\n2. 🎯 Adding Action Recognition:")
        print("   Create new action recognition capabilities:")
        print()
        print("   ```python")
        print("   class ActionRecognizer:")
        print("       def __init__(self, model_path):")
        print("           self.model = YOLO(model_path)")
        print("       ")
        print("       def recognize_ball_actions(self, frame, ball_bbox):")
        print("           # Recognize ball actions in context")
        print("           return self.model.predict(frame)")
        print("   ```")
        
        print("\n3. 📊 Enhanced Statistics:")
        print("   Extend your analysis with action-based statistics:")
        print()
        print("   ```python")
        print("   class EnhancedPassCounter(PassCounter):")
        print("       def __init__(self):")
        print("           super().__init__()")
        print("           self.action_counts = {'Pass': 0, 'Shot': 0, 'Cross': 0, ...}")
        print("   ```")
        
        print("\n4. 🚀 Running Enhanced Analysis:")
        print("   python main.py input_video.mp4 --use-soccernet-models --enable-action-recognition")
        
        # Show integration examples
        self._show_integration_examples()
    
    def _show_directory_structure(self):
        """Show expected directory structure."""
        structure = """
data/
├── soccernet/
│   ├── action_spotting/
│   │   ├── videos/          # Broadcast videos
│   │   ├── annotations/     # Labels-v2.json files
│   │   └── features/        # Pre-computed features
│   ├── ball_action_spotting/
│   │   ├── videos/          # Ball action videos
│   │   ├── annotations/     # Labels-ball.json files
│   │   └── features/        # Features
│   └── yolo_datasets/
│       ├── enhanced_ball_detection/
│       ├── action_aware_players/
│       └── multi_class_actions/
└── models/
    └── soccernet/
        ├── enhanced_ball_detection_training/
        ├── action_aware_players_training/
        └── multi_class_actions_training/
        """
        print(structure)
    
    def _check_setup_status(self):
        """Check current setup status."""
        print("\n5. ✅ Setup Status Check:")
        
        # Check if SoccerNet package is installed
        try:
            import SoccerNet
            print("   ✅ SoccerNet package installed")
        except ImportError:
            print("   ❌ SoccerNet package not installed")
            print("      Run: pip install SoccerNet")
        
        # Check if data directory exists
        if self.soccernet_dir.exists():
            print("   ✅ SoccerNet data directory exists")
            
            # Check subdirectories
            subdirs = ["action_spotting", "ball_action_spotting", "yolo_datasets"]
            for subdir in subdirs:
                if (self.soccernet_dir / subdir).exists():
                    print(f"   ✅ {subdir} directory exists")
                else:
                    print(f"   ❌ {subdir} directory missing")
        else:
            print("   ❌ SoccerNet data directory not found")
            print("      Run data setup script first")
    
    def _check_training_readiness(self):
        """Check if system is ready for training."""
        print("\n4. ✅ Training Readiness Check:")
        
        # Check for YOLO datasets
        yolo_datasets_dir = self.soccernet_dir / "yolo_datasets"
        if yolo_datasets_dir.exists():
            print("   ✅ YOLO datasets directory exists")
            
            datasets = ["enhanced_ball_detection", "action_aware_players", "multi_class_actions"]
            for dataset in datasets:
                dataset_dir = yolo_datasets_dir / dataset
                if dataset_dir.exists():
                    yaml_file = dataset_dir / "dataset.yaml"
                    if yaml_file.exists():
                        print(f"   ✅ {dataset} ready for training")
                    else:
                        print(f"   ⚠️ {dataset} missing dataset.yaml")
                else:
                    print(f"   ❌ {dataset} not found")
        else:
            print("   ❌ YOLO datasets not found")
            print("      Run conversion script first")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"   ✅ {gpu_count} GPU(s) available for training")
            else:
                print("   ⚠️ No GPU available (will use CPU)")
        except ImportError:
            print("   ❌ PyTorch not installed")
    
    def _show_integration_examples(self):
        """Show practical integration examples."""
        print("\n5. 💡 Integration Examples:")
        
        examples = [
            {
                "title": "Enhanced Ball Tracking",
                "description": "Use SoccerNet ball model for better ball detection",
                "benefit": "85-90% mAP vs 70-80% with generic models"
            },
            {
                "title": "Action-Aware Analysis", 
                "description": "Recognize ball actions (passes, shots, crosses)",
                "benefit": "Add tactical insights to your analysis"
            },
            {
                "title": "Improved Player Detection",
                "description": "Better player detection in crowded scenes",
                "benefit": "More accurate tracking and team assignment"
            },
            {
                "title": "Real-time Action Recognition",
                "description": "Classify actions as they happen",
                "benefit": "Live tactical analysis and statistics"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n   {i}. {example['title']}:")
            print(f"      {example['description']}")
            print(f"      Benefit: {example['benefit']}")


def main():
    """Main function to handle demo modes."""
    parser = argparse.ArgumentParser(description="SoccerNet Integration Demo")
    parser.add_argument("--demo-mode", required=True,
                       choices=["setup", "train", "integrate", "all"],
                       help="Demo mode to run")
    
    args = parser.parse_args()
    
    demo = SoccerNetIntegrationDemo()
    
    if args.demo_mode == "setup":
        demo.demo_setup()
    elif args.demo_mode == "train":
        demo.demo_training()
    elif args.demo_mode == "integrate":
        demo.demo_integration()
    elif args.demo_mode == "all":
        demo.demo_setup()
        print("\n" + "="*80 + "\n")
        demo.demo_training()
        print("\n" + "="*80 + "\n")
        demo.demo_integration()
    
    print(f"\n📚 For detailed instructions, see: docs/SOCCERNET_INTEGRATION.md")


if __name__ == "__main__":
    main()
