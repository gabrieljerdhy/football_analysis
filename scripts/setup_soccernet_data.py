#!/usr/bin/env python3
"""
SoccerNet Data Setup Script

This script downloads and organizes SoccerNet datasets for training YOLOv8 models.
It handles both Action Spotting and Ball Action Spotting datasets.

Usage:
    python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD
    python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD --dataset ball-action-spotting
    python scripts/setup_soccernet_data.py --password YOUR_NDA_PASSWORD --dataset action-spotting
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError:
    print("❌ SoccerNet package not found. Please install it with: pip install SoccerNet")
    sys.exit(1)


class SoccerNetDatasetManager:
    """Manages SoccerNet dataset downloads and organization for YOLOv8 training."""
    
    def __init__(self, local_directory: str = "data/soccernet"):
        """
        Initialize the dataset manager.
        
        Args:
            local_directory: Local directory to store SoccerNet data
        """
        self.local_directory = Path(local_directory)
        self.local_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize SoccerNet downloader
        self.downloader = SoccerNetDownloader(LocalDirectory=str(self.local_directory))
        
        # Create subdirectories for organized data
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create organized directory structure for different datasets."""
        directories = [
            "action_spotting/videos",
            "action_spotting/annotations", 
            "action_spotting/features",
            "ball_action_spotting/videos",
            "ball_action_spotting/annotations",
            "ball_action_spotting/features",
            "yolo_datasets/action_spotting",
            "yolo_datasets/ball_action_spotting",
            "yolo_datasets/enhanced_ball_detection",
            "yolo_datasets/action_aware_players"
        ]
        
        for directory in directories:
            (self.local_directory / directory).mkdir(parents=True, exist_ok=True)
    
    def set_password(self, password: str):
        """Set the NDA password for video downloads."""
        self.downloader.password = password
    
    def download_action_spotting_data(self, splits: List[str] = None, include_videos: bool = True):
        """
        Download SoccerNet Action Spotting dataset.
        
        Args:
            splits: List of splits to download ['train', 'valid', 'test', 'challenge']
            include_videos: Whether to download video files (requires NDA password)
        """
        if splits is None:
            splits = ["train", "valid", "test"]
        
        print(f"🏈 Downloading SoccerNet Action Spotting data for splits: {splits}")
        
        # Download annotations
        print("📋 Downloading annotations...")
        self.downloader.downloadGames(
            files=["Labels-v2.json"], 
            split=splits
        )
        
        # Download features (ResNet-152 and Baidu features)
        print("🧠 Downloading pre-computed features...")
        feature_files = [
            "1_ResNET_TF2_PCA512.npy", 
            "2_ResNET_TF2_PCA512.npy",
            "1_baidu_soccer_embeddings.npy", 
            "2_baidu_soccer_embeddings.npy"
        ]
        self.downloader.downloadGames(files=feature_files, split=splits)
        
        # Download videos if password is provided
        if include_videos and hasattr(self.downloader, 'password') and self.downloader.password:
            print("🎬 Downloading videos...")
            video_files = ["1_224p.mkv", "2_224p.mkv", "1_720p.mkv", "2_720p.mkv", "video.ini"]
            self.downloader.downloadGames(files=video_files, split=splits)
        elif include_videos:
            print("⚠️ No password provided. Skipping video downloads.")
            print("   To download videos, provide NDA password with --password argument")
    
    def download_ball_action_spotting_data(self, splits: List[str] = None, include_videos: bool = True):
        """
        Download SoccerNet Ball Action Spotting dataset.
        
        Args:
            splits: List of splits to download ['train', 'valid', 'test', 'challenge']
            include_videos: Whether to download video files (requires NDA password)
        """
        if splits is None:
            splits = ["train", "valid", "test"]
        
        print(f"⚽ Downloading SoccerNet Ball Action Spotting data for splits: {splits}")
        
        # Download ball action spotting data using the task-specific method
        if hasattr(self.downloader, 'password') and self.downloader.password:
            print("📦 Downloading ball action spotting dataset...")
            self.downloader.downloadDataTask(
                task="spotting-ball-2024", 
                split=splits, 
                password=self.downloader.password
            )
        else:
            print("⚠️ No password provided. Cannot download ball action spotting data.")
            print("   Ball action spotting requires NDA password.")
            return
    
    def organize_downloaded_data(self):
        """Organize downloaded data into structured directories."""
        print("📁 Organizing downloaded data...")
        
        # Move action spotting data
        self._organize_action_spotting_data()
        
        # Move ball action spotting data  
        self._organize_ball_action_spotting_data()
        
        print("✅ Data organization complete!")
    
    def _organize_action_spotting_data(self):
        """Organize action spotting data into structured directories."""
        source_dir = self.local_directory
        target_dir = self.local_directory / "action_spotting"
        
        # Find and move annotation files
        for labels_file in source_dir.rglob("Labels-v2.json"):
            relative_path = labels_file.relative_to(source_dir)
            target_path = target_dir / "annotations" / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                labels_file.rename(target_path)
        
        # Find and move video files
        for video_file in source_dir.rglob("*.mkv"):
            relative_path = video_file.relative_to(source_dir)
            target_path = target_dir / "videos" / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                video_file.rename(target_path)
        
        # Find and move feature files
        for feature_file in source_dir.rglob("*.npy"):
            if "ResNET" in feature_file.name or "baidu" in feature_file.name:
                relative_path = feature_file.relative_to(source_dir)
                target_path = target_dir / "features" / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.exists():
                    feature_file.rename(target_path)
    
    def _organize_ball_action_spotting_data(self):
        """Organize ball action spotting data into structured directories."""
        source_dir = self.local_directory
        target_dir = self.local_directory / "ball_action_spotting"
        
        # Find and move ball action annotation files
        for labels_file in source_dir.rglob("Labels-ball.json"):
            relative_path = labels_file.relative_to(source_dir)
            target_path = target_dir / "annotations" / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                labels_file.rename(target_path)
        
        # Ball action spotting videos are typically in specific directories
        # Look for ball action specific video directories
        for video_dir in source_dir.rglob("*ball*"):
            if video_dir.is_dir():
                for video_file in video_dir.rglob("*.mkv"):
                    relative_path = video_file.relative_to(source_dir)
                    target_path = target_dir / "videos" / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not target_path.exists():
                        video_file.rename(target_path)
    
    def get_dataset_info(self):
        """Get information about downloaded datasets."""
        info = {
            "action_spotting": {
                "annotations": len(list((self.local_directory / "action_spotting/annotations").rglob("*.json"))),
                "videos": len(list((self.local_directory / "action_spotting/videos").rglob("*.mkv"))),
                "features": len(list((self.local_directory / "action_spotting/features").rglob("*.npy")))
            },
            "ball_action_spotting": {
                "annotations": len(list((self.local_directory / "ball_action_spotting/annotations").rglob("*.json"))),
                "videos": len(list((self.local_directory / "ball_action_spotting/videos").rglob("*.mkv"))),
                "features": len(list((self.local_directory / "ball_action_spotting/features").rglob("*.npy")))
            }
        }
        return info


def main():
    """Main function to handle command line arguments and execute data setup."""
    parser = argparse.ArgumentParser(description="Setup SoccerNet datasets for YOLOv8 training")
    parser.add_argument("--password", type=str, help="NDA password for video downloads")
    parser.add_argument("--dataset", choices=["action-spotting", "ball-action-spotting", "both"], 
                       default="both", help="Which dataset to download")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                       choices=["train", "valid", "test", "challenge"],
                       help="Dataset splits to download")
    parser.add_argument("--no-videos", action="store_true", 
                       help="Skip video downloads (annotations and features only)")
    parser.add_argument("--local-dir", type=str, default="data/soccernet",
                       help="Local directory to store SoccerNet data")
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    manager = SoccerNetDatasetManager(args.local_dir)
    
    # Set password if provided
    if args.password:
        manager.set_password(args.password)
    
    # Download datasets based on selection
    include_videos = not args.no_videos
    
    if args.dataset in ["action-spotting", "both"]:
        manager.download_action_spotting_data(args.splits, include_videos)
    
    if args.dataset in ["ball-action-spotting", "both"]:
        manager.download_ball_action_spotting_data(args.splits, include_videos)
    
    # Organize downloaded data
    manager.organize_downloaded_data()
    
    # Print dataset information
    info = manager.get_dataset_info()
    print("\n📊 Dataset Summary:")
    print("=" * 50)
    for dataset_name, dataset_info in info.items():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        for data_type, count in dataset_info.items():
            print(f"  {data_type.title()}: {count} files")
    
    print(f"\n✅ SoccerNet data setup complete!")
    print(f"📁 Data stored in: {manager.local_directory}")


if __name__ == "__main__":
    main()
