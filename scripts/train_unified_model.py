#!/usr/bin/env python3
"""
Unified Football Model Training Script

This script trains the unified multi-task YOLO model that replaces the current
three-model approach (player detection, ball detection, field keypoints).

Usage:
    python scripts/train_unified_model.py --prepare-data
    python scripts/train_unified_model.py --train --epochs 100
    python scripts/train_unified_model.py --validate --model data/models/unified_football_detector.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.unified_model_trainer import UnifiedModelTrainer
from models.unified_model_adapter import create_unified_adapter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train unified football detection model")
    
    # Main actions
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Prepare unified dataset from separate datasets")
    parser.add_argument("--train", action="store_true", 
                       help="Train unified model")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate trained model")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Benchmark unified model vs separate models")
    
    # Data paths
    parser.add_argument("--player-data", type=str, 
                       default="data/datasets/player_detection",
                       help="Path to player detection dataset")
    parser.add_argument("--ball-data", type=str, 
                       default="data/datasets/ball_detection", 
                       help="Path to ball detection dataset")
    parser.add_argument("--keypoint-data", type=str, 
                       default="data/datasets/keypoint_detection",
                       help="Path to keypoint detection dataset")
    parser.add_argument("--output-data", type=str, 
                       default="data/datasets/unified_football",
                       help="Output path for unified dataset")
    
    # Training parameters
    parser.add_argument("--base-model", type=str, default="yolov8m.pt",
                       help="Base YOLO model for transfer learning")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                       help="Training device (cuda/cpu)")
    
    # Model paths
    parser.add_argument("--model", type=str, 
                       default="data/models/unified_football_detector.pt",
                       help="Path to model for validation/benchmarking")
    parser.add_argument("--output-dir", type=str, 
                       default="data/models/unified",
                       help="Output directory for trained models")
    
    # Options
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def prepare_unified_dataset(args):
    """Prepare unified dataset from separate task datasets."""
    print("🔄 Preparing Unified Dataset")
    print("=" * 50)
    
    # Check if source datasets exist
    datasets = {
        'player': args.player_data,
        'ball': args.ball_data,
        'keypoint': args.keypoint_data
    }
    
    missing_datasets = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            missing_datasets.append(f"{name}: {path}")
    
    if missing_datasets:
        print("❌ Missing datasets:")
        for dataset in missing_datasets:
            print(f"   {dataset}")
        print("\n💡 To create datasets, you can:")
        print("   1. Use existing YOLO datasets in the specified paths")
        print("   2. Convert SoccerNet data using scripts/convert_soccernet_to_yolo.py")
        print("   3. Create custom datasets following YOLO format")
        return False
    
    # Initialize trainer
    trainer = UnifiedModelTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Prepare unified dataset
    try:
        dataset_config = trainer.prepare_unified_dataset(
            player_data_path=args.player_data,
            ball_data_path=args.ball_data,
            keypoint_data_path=args.keypoint_data,
            output_path=args.output_data
        )
        
        print(f"✅ Unified dataset prepared successfully")
        print(f"   Dataset config: {dataset_config}")
        print(f"   Output directory: {args.output_data}")
        
        return dataset_config
        
    except Exception as e:
        print(f"❌ Failed to prepare dataset: {e}")
        return False


def train_unified_model(args, dataset_config=None):
    """Train unified football detection model."""
    print("🚀 Training Unified Model")
    print("=" * 50)
    
    # Use provided dataset config or default path
    if dataset_config is None:
        dataset_config = os.path.join(args.output_data, "dataset.yaml")
    
    if not os.path.exists(dataset_config):
        print(f"❌ Dataset config not found: {dataset_config}")
        print("💡 Run with --prepare-data first to create the dataset")
        return False
    
    # Initialize trainer
    trainer = UnifiedModelTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Update training configuration
    trainer.training_config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    print(f"📋 Training Configuration:")
    print(f"   Base model: {args.base_model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"   Dataset: {dataset_config}")
    
    # Start training
    try:
        start_time = time.time()
        
        model_path = trainer.train_unified_model(
            dataset_config=dataset_config,
            model_name="unified_football_detector",
            resume=args.resume
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed successfully")
        print(f"   Training time: {training_time/3600:.2f} hours")
        print(f"   Model saved: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return False


def validate_unified_model(args):
    """Validate trained unified model."""
    print("🔍 Validating Unified Model")
    print("=" * 50)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return False
    
    dataset_config = os.path.join(args.output_data, "dataset.yaml")
    if not os.path.exists(dataset_config):
        print(f"❌ Dataset config not found: {dataset_config}")
        return False
    
    # Initialize trainer for validation
    trainer = UnifiedModelTrainer(device=args.device)
    
    try:
        metrics = trainer.validate_model(args.model, dataset_config)
        
        print(f"✅ Validation completed")
        print(f"   Model: {args.model}")
        print(f"   mAP@0.5: {metrics['mAP50']:.3f}")
        print(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1-Score: {metrics['f1']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def benchmark_unified_model(args):
    """Benchmark unified model against separate models."""
    print("⚡ Benchmarking Unified Model")
    print("=" * 50)
    
    if not os.path.exists(args.model):
        print(f"❌ Unified model not found: {args.model}")
        return False
    
    try:
        # Create test frames
        import numpy as np
        test_frames = [
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(50)
        ]
        
        print(f"🧪 Testing with {len(test_frames)} frames")
        
        # Test unified model
        print("\n1️⃣ Testing Unified Model...")
        unified_adapter = create_unified_adapter(args.model, args.device)
        
        start_time = time.time()
        unified_results = unified_adapter.get_object_tracks(test_frames)
        unified_time = time.time() - start_time
        
        unified_metrics = unified_adapter.get_performance_metrics()
        
        print(f"   Processing time: {unified_time:.2f}s")
        print(f"   FPS: {len(test_frames)/unified_time:.1f}")
        print(f"   Model type: {unified_metrics.get('model_type', 'unknown')}")
        
        # Test separate models (if available)
        print("\n2️⃣ Testing Separate Models...")
        try:
            from src.trackers.tracker import Tracker
            from src.goal_detection.field_keypoints_detector import FieldKeypointsDetector
            
            # Initialize separate models
            tracker = Tracker(
                "data/models/best_player_detect.pt",
                ball_model_path="data/models/best_ball_latest.pt",
                enable_enhanced_ball_detection=True,
                device=args.device
            )
            keypoint_detector = FieldKeypointsDetector(
                "data/models/best_field_keypoint.pt",
                device=args.device
            )
            
            start_time = time.time()
            separate_results = tracker.get_object_tracks(test_frames)
            
            # Process keypoints for each frame
            for frame in test_frames[:10]:  # Sample frames for keypoint detection
                _ = keypoint_detector.detect_keypoints(frame)
            
            separate_time = time.time() - start_time
            
            print(f"   Processing time: {separate_time:.2f}s")
            print(f"   FPS: {len(test_frames)/separate_time:.1f}")
            
            # Calculate performance improvement
            speed_improvement = (separate_time - unified_time) / separate_time * 100
            fps_improvement = (len(test_frames)/unified_time) / (len(test_frames)/separate_time) - 1
            
            print(f"\n📊 Performance Comparison:")
            print(f"   Speed improvement: {speed_improvement:.1f}%")
            print(f"   FPS improvement: {fps_improvement*100:.1f}%")
            print(f"   Time saved: {separate_time - unified_time:.2f}s")
            
        except Exception as e:
            print(f"   ⚠️ Could not test separate models: {e}")
            print(f"   Using unified model performance metrics only")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🏈 Unified Football Model Training Pipeline")
    print("=" * 60)
    
    success = True
    dataset_config = None
    
    # Prepare data if requested
    if args.prepare_data:
        dataset_config = prepare_unified_dataset(args)
        if not dataset_config:
            success = False
    
    # Train model if requested
    if args.train and success:
        model_path = train_unified_model(args, dataset_config)
        if model_path:
            args.model = model_path  # Use newly trained model for validation
        else:
            success = False
    
    # Validate model if requested
    if args.validate and success:
        validation_results = validate_unified_model(args)
        if not validation_results:
            success = False
    
    # Benchmark model if requested
    if args.benchmark and success:
        benchmark_results = benchmark_unified_model(args)
        if not benchmark_results:
            success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All operations completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Integrate unified model into main pipeline")
        print("   2. Update main.py to use UnifiedModelAdapter")
        print("   3. Run performance tests on real football videos")
        print("   4. Monitor accuracy improvements in goal detection")
    else:
        print("❌ Some operations failed. Check the output above for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
