#!/usr/bin/env python3
"""
S3 Video Analysis Demo

This script demonstrates how to use the football analysis system with S3 video inputs.
It shows different ways to configure AWS credentials and process videos stored in S3.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import S3VideoHandler, is_s3_uri, parse_s3_uri


def demo_s3_uri_validation():
    """Demonstrate S3 URI validation and parsing."""
    print("🔍 S3 URI Validation Demo")
    print("=" * 50)
    
    test_uris = [
        "s3://my-bucket/videos/match.mp4",
        "s3://football-videos/2023/championship/final.avi",
        "s3://invalid-uri",
        "local/file/path.mp4",
        "https://example.com/video.mp4",
    ]
    
    for uri in test_uris:
        print(f"\nTesting: {uri}")
        if is_s3_uri(uri):
            try:
                bucket, key = parse_s3_uri(uri)
                print(f"  ✅ Valid S3 URI")
                print(f"     Bucket: {bucket}")
                print(f"     Key: {key}")
            except ValueError as e:
                print(f"  ❌ Invalid S3 URI: {e}")
        else:
            print(f"  ℹ️  Not an S3 URI")


def demo_s3_authentication():
    """Demonstrate different S3 authentication methods."""
    print("\n🔐 S3 Authentication Demo")
    print("=" * 50)
    
    print("\n1. Environment Variables Method:")
    print("   export AWS_ACCESS_KEY_ID=your_access_key")
    print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("   export AWS_DEFAULT_REGION=us-east-1")
    
    # Test environment variable authentication
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        print("   ✅ AWS credentials found in environment")
        try:
            handler = S3VideoHandler()
            if handler.s3_client:
                print("   ✅ S3 client created successfully")
            else:
                print("   ❌ Failed to create S3 client")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️  AWS credentials not found in environment")
    
    print("\n2. AWS Profile Method:")
    print("   aws configure --profile my-profile")
    print("   python main.py --input s3://bucket/video.mp4 --aws-profile my-profile")
    
    print("\n3. Command Line Arguments Method:")
    print("   python main.py --input s3://bucket/video.mp4 \\")
    print("     --aws-access-key-id YOUR_KEY \\")
    print("     --aws-secret-access-key YOUR_SECRET \\")
    print("     --aws-region us-west-2")


def demo_s3_video_processing():
    """Demonstrate S3 video processing workflow."""
    print("\n🎬 S3 Video Processing Demo")
    print("=" * 50)
    
    # Example S3 URIs (these are just examples, not real videos)
    example_s3_uris = [
        "s3://football-analysis-videos/matches/premier-league/2023/match-001.mp4",
        "s3://sports-content/soccer/world-cup/final.avi",
        "s3://my-bucket/training-videos/tactics-demo.mov",
    ]
    
    print("\nExample S3 video processing commands:")
    
    for i, s3_uri in enumerate(example_s3_uris, 1):
        print(f"\n{i}. Basic analysis:")
        print(f"   python main.py --input {s3_uri}")
        
        print(f"\n   Memory-efficient analysis:")
        print(f"   python main.py --input {s3_uri} --memory-efficient")
        
        print(f"\n   Full analysis with all features:")
        print(f"   python main.py --input {s3_uri} \\")
        print(f"     --memory-efficient \\")
        print(f"     --enable-camera-movement \\")
        print(f"     --enable-speed-distance \\")
        print(f"     --batch-size 30")


def demo_error_scenarios():
    """Demonstrate common error scenarios and troubleshooting."""
    print("\n🚨 Common Error Scenarios & Troubleshooting")
    print("=" * 50)
    
    scenarios = [
        {
            "error": "❌ AWS credentials not found",
            "cause": "No AWS credentials configured",
            "solutions": [
                "Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
                "Run 'aws configure' to set up credentials",
                "Use command line arguments: --aws-access-key-id, --aws-secret-access-key",
                "Use AWS profile: --aws-profile profile-name"
            ]
        },
        {
            "error": "❌ S3 bucket not found",
            "cause": "Bucket name is incorrect or doesn't exist",
            "solutions": [
                "Check bucket name spelling",
                "Verify bucket exists in the specified region",
                "Ensure you have access to the bucket"
            ]
        },
        {
            "error": "❌ S3 object not found",
            "cause": "Video file path is incorrect or file doesn't exist",
            "solutions": [
                "Check the object key (file path) is correct",
                "Verify the file exists in the bucket",
                "Check if file was moved or deleted"
            ]
        },
        {
            "error": "❌ Access denied to S3 object",
            "cause": "Insufficient permissions to access the file",
            "solutions": [
                "Check AWS credentials have read permissions",
                "Verify bucket policy allows access",
                "Check if object is encrypted with different key"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔸 {scenario['error']}")
        print(f"   Cause: {scenario['cause']}")
        print("   Solutions:")
        for solution in scenario['solutions']:
            print(f"   • {solution}")


def demo_performance_tips():
    """Demonstrate performance optimization tips for S3 videos."""
    print("\n⚡ Performance Optimization Tips")
    print("=" * 50)
    
    tips = [
        {
            "tip": "Use memory-efficient processing",
            "command": "python main.py --input s3://bucket/video.mp4 --memory-efficient",
            "benefit": "Reduces memory usage for large videos"
        },
        {
            "tip": "Adjust batch size based on video size",
            "command": "python main.py --input s3://bucket/large-video.mp4 --batch-size 20",
            "benefit": "Smaller batches for very large videos"
        },
        {
            "tip": "Choose optimal AWS region",
            "command": "python main.py --input s3://bucket/video.mp4 --aws-region us-west-2",
            "benefit": "Reduce network latency by using nearby region"
        },
        {
            "tip": "Check video info before processing",
            "command": "python main.py --input s3://bucket/video.mp4 --check-video-info",
            "benefit": "Verify video requirements without full download"
        }
    ]
    
    for tip in tips:
        print(f"\n💡 {tip['tip']}")
        print(f"   Command: {tip['command']}")
        print(f"   Benefit: {tip['benefit']}")


def main():
    """Run all S3 video analysis demos."""
    print("🚀 Football Analysis S3 Video Input Demo")
    print("=" * 60)
    
    demo_s3_uri_validation()
    demo_s3_authentication()
    demo_s3_video_processing()
    demo_error_scenarios()
    demo_performance_tips()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\n📚 For more information, see the README.md file.")
    print("🔗 S3 Video Input Documentation: https://github.com/your-repo#s3-video-input-setup")


if __name__ == "__main__":
    main()
