#!/usr/bin/env python3
"""
Multi-Provider Object Storage Demo

This script demonstrates the football analysis system's support for multiple
object storage providers including AWS S3, DigitalOcean Spaces, Google Cloud Storage,
Azure Blob Storage, and MinIO.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    MultiStorageHandler,
    StorageProvider,
    detect_storage_provider,
    is_object_storage_uri,
    parse_storage_uri,
)


def demo_provider_detection():
    """Demonstrate automatic storage provider detection."""
    print("🔍 Storage Provider Detection Demo")
    print("=" * 50)
    
    test_uris = [
        "s3://my-bucket/videos/match.mp4",
        "gs://sports-videos/championship/final.avi",
        "azure://account.blob.core.windows.net/videos/match.mp4",
        "spaces://bucket.nyc3.digitaloceanspaces.com/videos/match.mp4",
        "minio://minio.example.com/bucket/videos/match.mp4",
        "https://bucket.fra1.digitaloceanspaces.com/videos/match.mp4",
        "/local/path/video.mp4",  # Not object storage
        "http://example.com/video.mp4",  # Not object storage
    ]
    
    for uri in test_uris:
        print(f"\nTesting URI: {uri}")
        
        if is_object_storage_uri(uri):
            provider, is_valid = detect_storage_provider(uri)
            print(f"  ✅ Detected: {provider}")
            
            try:
                parsed = parse_storage_uri(uri)
                print(f"  📋 Parsed components:")
                for key, value in parsed.items():
                    if value is not None:
                        print(f"     {key}: {value}")
            except Exception as e:
                print(f"  ❌ Parse error: {e}")
        else:
            print(f"  ℹ️  Not an object storage URI")


def demo_usage_examples():
    """Demonstrate usage examples for each storage provider."""
    print("\n🚀 Usage Examples")
    print("=" * 50)
    
    providers = [
        {
            "name": "AWS S3",
            "uri": "s3://my-bucket/videos/match.mp4",
            "auth_examples": [
                "# Environment variables",
                "export AWS_ACCESS_KEY_ID=your_access_key",
                "export AWS_SECRET_ACCESS_KEY=your_secret_key",
                "",
                "# Command line",
                "python main.py --input s3://bucket/video.mp4 \\",
                "  --aws-access-key-id YOUR_KEY \\",
                "  --aws-secret-access-key YOUR_SECRET"
            ]
        },
        {
            "name": "Google Cloud Storage",
            "uri": "gs://sports-videos/match.mp4",
            "auth_examples": [
                "# Service account file",
                "export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                "",
                "# Command line",
                "python main.py --input gs://bucket/video.mp4 \\",
                "  --gcs-service-account-path /path/to/service-account.json"
            ]
        },
        {
            "name": "Azure Blob Storage",
            "uri": "azure://account.blob.core.windows.net/videos/match.mp4",
            "auth_examples": [
                "# Environment variables",
                "export AZURE_STORAGE_ACCOUNT=your_account",
                "export AZURE_STORAGE_KEY=your_key",
                "",
                "# Command line",
                "python main.py --input azure://account.blob.core.windows.net/container/video.mp4 \\",
                "  --azure-account-name ACCOUNT \\",
                "  --azure-account-key KEY"
            ]
        },
        {
            "name": "DigitalOcean Spaces",
            "uri": "spaces://bucket.nyc3.digitaloceanspaces.com/videos/match.mp4",
            "auth_examples": [
                "# Environment variables",
                "export DO_SPACES_ACCESS_KEY_ID=your_key",
                "export DO_SPACES_SECRET_ACCESS_KEY=your_secret",
                "",
                "# Command line",
                "python main.py --input spaces://bucket.region.digitaloceanspaces.com/video.mp4 \\",
                "  --spaces-access-key-id-input YOUR_KEY \\",
                "  --spaces-secret-access-key-input YOUR_SECRET"
            ]
        },
        {
            "name": "MinIO",
            "uri": "minio://minio.example.com/bucket/videos/match.mp4",
            "auth_examples": [
                "# Environment variables",
                "export MINIO_ENDPOINT=http://minio.example.com",
                "export MINIO_ACCESS_KEY=your_key",
                "export MINIO_SECRET_KEY=your_secret",
                "",
                "# Command line",
                "python main.py --input minio://endpoint/bucket/video.mp4 \\",
                "  --minio-endpoint http://minio.example.com \\",
                "  --minio-access-key YOUR_KEY \\",
                "  --minio-secret-key YOUR_SECRET"
            ]
        }
    ]
    
    for provider in providers:
        print(f"\n📦 {provider['name']}")
        print(f"   Example URI: {provider['uri']}")
        print(f"   Authentication:")
        for line in provider['auth_examples']:
            print(f"   {line}")


def demo_command_examples():
    """Demonstrate complete command examples."""
    print("\n💻 Complete Command Examples")
    print("=" * 50)
    
    examples = [
        {
            "title": "Basic video analysis with S3",
            "command": "python main.py --input s3://my-bucket/match.mp4"
        },
        {
            "title": "Memory-efficient processing with Google Cloud Storage",
            "command": "python main.py --input gs://sports-videos/large-match.mp4 --memory-efficient"
        },
        {
            "title": "Full analysis with Azure Blob Storage",
            "command": """python main.py --input azure://account.blob.core.windows.net/videos/match.mp4 \\
  --memory-efficient \\
  --enable-camera-movement \\
  --enable-speed-distance \\
  --azure-account-name myaccount \\
  --azure-account-key mykey"""
        },
        {
            "title": "DigitalOcean Spaces with custom region",
            "command": """python main.py --input spaces://bucket.fra1.digitaloceanspaces.com/match.mp4 \\
  --spaces-access-key-id-input YOUR_KEY \\
  --spaces-secret-access-key-input YOUR_SECRET \\
  --spaces-region-input fra1"""
        },
        {
            "title": "MinIO with custom endpoint",
            "command": """python main.py --input minio://minio.company.com/videos/match.mp4 \\
  --minio-endpoint http://minio.company.com \\
  --minio-access-key company_key \\
  --minio-secret-key company_secret"""
        },
        {
            "title": "Check video info before processing",
            "command": "python main.py --input gs://bucket/video.mp4 --check-video-info"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")


def demo_error_scenarios():
    """Demonstrate error handling and troubleshooting."""
    print("\n🚨 Error Handling & Troubleshooting")
    print("=" * 50)
    
    scenarios = [
        {
            "error": "❌ google-cloud-storage is required for GCS support",
            "cause": "Missing Google Cloud Storage dependencies",
            "solution": "pip install google-cloud-storage"
        },
        {
            "error": "❌ azure-storage-blob is required for Azure support",
            "cause": "Missing Azure Blob Storage dependencies",
            "solution": "pip install azure-storage-blob"
        },
        {
            "error": "❌ AWS credentials not found",
            "cause": "No AWS credentials configured",
            "solutions": [
                "Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
                "Run 'aws configure' to set up credentials",
                "Use command line: --aws-access-key-id, --aws-secret-access-key"
            ]
        },
        {
            "error": "❌ DigitalOcean Spaces credentials not found",
            "cause": "No Spaces credentials configured",
            "solutions": [
                "Set environment variables: DO_SPACES_ACCESS_KEY_ID, DO_SPACES_SECRET_ACCESS_KEY",
                "Use command line: --spaces-access-key-id-input, --spaces-secret-access-key-input"
            ]
        },
        {
            "error": "❌ MinIO credentials or endpoint not found",
            "cause": "Missing MinIO configuration",
            "solutions": [
                "Set environment variables: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY",
                "Use command line: --minio-endpoint, --minio-access-key, --minio-secret-key"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔸 {scenario['error']}")
        print(f"   Cause: {scenario['cause']}")
        if 'solution' in scenario:
            print(f"   Solution: {scenario['solution']}")
        elif 'solutions' in scenario:
            print("   Solutions:")
            for solution in scenario['solutions']:
                print(f"   • {solution}")


def demo_performance_tips():
    """Demonstrate performance optimization tips."""
    print("\n⚡ Performance Optimization Tips")
    print("=" * 50)
    
    tips = [
        {
            "tip": "Use memory-efficient processing for large videos",
            "command": "python main.py --input gs://bucket/large-video.mp4 --memory-efficient",
            "benefit": "Reduces memory usage by processing video in batches"
        },
        {
            "tip": "Adjust batch size based on available memory",
            "command": "python main.py --input s3://bucket/video.mp4 --batch-size 20",
            "benefit": "Smaller batches for systems with limited memory"
        },
        {
            "tip": "Choose storage provider closest to your location",
            "command": "# Use regional endpoints for better performance",
            "benefit": "Reduces network latency and download times"
        },
        {
            "tip": "Check video info before full processing",
            "command": "python main.py --input azure://account.blob.core.windows.net/container/video.mp4 --check-video-info",
            "benefit": "Verify video requirements without downloading entire file"
        },
        {
            "tip": "Use appropriate credentials method",
            "command": "# Environment variables are fastest, service accounts are most secure",
            "benefit": "Optimizes authentication performance and security"
        }
    ]
    
    for tip in tips:
        print(f"\n💡 {tip['tip']}")
        print(f"   Command: {tip['command']}")
        print(f"   Benefit: {tip['benefit']}")


def demo_supported_formats():
    """Demonstrate supported URI formats."""
    print("\n📋 Supported URI Formats")
    print("=" * 50)
    
    formats = [
        {
            "provider": "AWS S3",
            "formats": [
                "s3://bucket-name/path/to/video.mp4",
                "s3://my-bucket/videos/match.avi",
                "s3://sports-data/2023/championship/final.mov"
            ]
        },
        {
            "provider": "Google Cloud Storage",
            "formats": [
                "gs://bucket-name/path/to/video.mp4",
                "gs://sports-videos/match.avi",
                "gs://ml-datasets/football/training-video.mov"
            ]
        },
        {
            "provider": "Azure Blob Storage",
            "formats": [
                "azure://account.blob.core.windows.net/container/video.mp4",
                "https://account.blob.core.windows.net/videos/match.avi"
            ]
        },
        {
            "provider": "DigitalOcean Spaces",
            "formats": [
                "spaces://bucket.region.digitaloceanspaces.com/video.mp4",
                "https://bucket.nyc3.digitaloceanspaces.com/videos/match.avi"
            ]
        },
        {
            "provider": "MinIO",
            "formats": [
                "minio://endpoint/bucket/video.mp4",
                "minio://localhost:9000/videos/match.avi",
                "minio://minio.company.com/sports/match.mov"
            ]
        }
    ]
    
    for provider_info in formats:
        print(f"\n📦 {provider_info['provider']}:")
        for format_example in provider_info['formats']:
            print(f"   • {format_example}")


def main():
    """Run all multi-storage demos."""
    print("🚀 Football Analysis Multi-Storage Demo")
    print("=" * 60)
    
    demo_provider_detection()
    demo_usage_examples()
    demo_command_examples()
    demo_error_scenarios()
    demo_performance_tips()
    demo_supported_formats()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\n📚 For more information:")
    print("• README.md - Complete documentation")
    print("• tests/test_multi_storage.py - Unit tests")
    print("• examples/s3_video_analysis_demo.py - S3-specific examples")


if __name__ == "__main__":
    main()
