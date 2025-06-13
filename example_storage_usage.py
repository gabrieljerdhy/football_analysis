#!/usr/bin/env python3
"""
Example script demonstrating how to use the DigitalOcean Spaces storage functionality
with the football analysis project.

This script shows different ways to configure and use the storage features.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def example_1_dotenv_file():
    """
    Example 1: Using .env file for configuration (Recommended)
    """
    print("🔧 Example 1: Using .env File (Recommended)")
    print("=" * 50)

    print("📝 Step 1: Create a .env file in your project root:")
    print("cp .env.example .env")
    print("\n📝 Step 2: Edit the .env file with your credentials:")
    print("DO_SPACES_ACCESS_KEY_ID=your_actual_access_key_id")
    print("DO_SPACES_SECRET_ACCESS_KEY=your_actual_secret_access_key")
    print("DO_SPACES_BUCKET=your_actual_bucket_name")
    print("DO_SPACES_REGION=nyc3")

    print("\n📝 Step 3: Run the analysis with upload:")
    print("python main.py --input video.mp4 --upload-to-spaces")

    # Test the uploader creation
    from utils.storage_utils import create_uploader_from_env

    uploader = create_uploader_from_env()
    if uploader:
        print("\n✅ Uploader created successfully from .env file")

        # Test connection
        bucket_name = os.getenv("DO_SPACES_BUCKET")
        if bucket_name and uploader.test_connection(bucket_name):
            print("✅ Connection test successful")
        else:
            print("❌ Connection test failed (expected with example credentials)")
    else:
        print("\n❌ Failed to create uploader (expected with example credentials)")


def example_2_environment_variables():
    """
    Example 2: Using environment variables for configuration
    """
    print("\n🔧 Example 2: Using Environment Variables")
    print("=" * 50)

    print("📝 To use environment variables, run:")
    print("export DO_SPACES_ACCESS_KEY_ID='your_access_key_id'")
    print("export DO_SPACES_SECRET_ACCESS_KEY='your_secret_access_key'")
    print("export DO_SPACES_BUCKET='your_bucket_name'")
    print("python main.py --input video.mp4 --upload-to-spaces")


def example_3_command_line_arguments():
    """
    Example 2: Using command line arguments
    """
    print("\n🔧 Example 2: Using Command Line Arguments")
    print("=" * 50)

    print("📝 To use command line arguments, run:")
    print("python main.py --input video.mp4 \\")
    print("    --upload-to-spaces \\")
    print("    --spaces-access-key-id 'your_access_key_id' \\")
    print("    --spaces-secret-access-key 'your_secret_access_key' \\")
    print("    --spaces-bucket 'your_bucket_name' \\")
    print("    --spaces-region 'nyc3' \\")
    print("    --spaces-folder-prefix 'football_analysis'")


def example_4_csv_only_upload():
    """
    Example 3: Upload only CSV files (faster)
    """
    print("\n🔧 Example 3: Upload CSV Files Only")
    print("=" * 50)

    print("📝 To upload only CSV files (skip video upload), run:")
    print("python main.py --input video.mp4 \\")
    print("    --upload-to-spaces \\")
    print("    --upload-csv-only \\")
    print("    --spaces-bucket 'your_bucket_name'")


def example_5_memory_efficient_with_storage():
    """
    Example 4: Memory-efficient processing with storage
    """
    print("\n🔧 Example 4: Memory-Efficient Processing with Storage")
    print("=" * 50)

    print("📝 For large videos with storage upload, run:")
    print("python main.py --input large_video.mp4 \\")
    print("    --memory-efficient \\")
    print("    --batch-size 30 \\")
    print("    --upload-to-spaces \\")
    print("    --spaces-bucket 'your_bucket_name'")


def example_6_direct_api_usage():
    """
    Example 5: Direct API usage for custom integrations
    """
    print("\n🔧 Example 5: Direct API Usage")
    print("=" * 50)

    try:
        from utils.storage_utils import create_uploader_from_args

        # Create uploader directly
        uploader = create_uploader_from_args(
            access_key_id="your_access_key_id",
            secret_access_key="your_secret_access_key",
            region="nyc3",
            bucket_name="your_bucket_name",
        )

        if uploader:
            print("✅ Uploader created successfully")

            # Example: Upload a single file
            # success = uploader.upload_file(
            #     local_file_path='output/video_team_stats.csv',
            #     remote_key='football_analysis/csv/video_team_stats.csv',
            #     bucket_name='your_bucket_name'
            # )

            # Example: Upload CSV files
            # results = uploader.upload_csv_files(
            #     team_csv_path='output/video_team_stats.csv',
            #     player_csv_path='output/video_player_stats.csv',
            #     video_name='video',
            #     bucket_name='your_bucket_name'
            # )

            # Example: Upload video
            # success = uploader.upload_video(
            #     video_path='output_videos/video_output.avi',
            #     video_name='video',
            #     bucket_name='your_bucket_name'
            # )

            print("📝 See commented code above for direct API usage examples")
        else:
            print("❌ Failed to create uploader")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure boto3 is installed: pip install boto3")


def example_7_folder_organization():
    """
    Example 6: Custom folder organization
    """
    print("\n🔧 Example 6: Custom Folder Organization")
    print("=" * 50)

    print("📝 To organize files in custom folders, run:")
    print("python main.py --input video.mp4 \\")
    print("    --upload-to-spaces \\")
    print("    --spaces-folder-prefix 'my_project/analysis_2024' \\")
    print("    --spaces-bucket 'your_bucket_name'")

    print("\n📁 This will create the following structure in your bucket:")
    print("my_project/analysis_2024/csv/video_team_stats.csv")
    print("my_project/analysis_2024/csv/video_player_stats.csv")
    print("my_project/analysis_2024/videos/video_output.avi")


def main():
    """
    Run all examples
    """
    print("🏈 Football Analysis - DigitalOcean Spaces Storage Examples")
    print("=" * 60)

    print("\n⚠️  Note: Replace the example credentials in .env with your actual")
    print("   DigitalOcean Spaces credentials.")

    example_1_dotenv_file()
    example_2_environment_variables()
    example_3_command_line_arguments()
    example_4_csv_only_upload()
    example_5_memory_efficient_with_storage()
    example_6_direct_api_usage()
    example_7_folder_organization()

    print("\n🔗 Additional Information:")
    print("- DigitalOcean Spaces is S3-compatible storage")
    print("- Default region is 'nyc3' but you can use any DO region")
    print("- Files are organized in folders: csv/ and videos/")
    print("- Upload progress and status are shown during processing")
    print("- Both regular and memory-efficient modes support storage")

    print("\n📚 For more information:")
    print("- DigitalOcean Spaces: https://www.digitalocean.com/products/spaces/")
    print(
        "- boto3 documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/"
    )


if __name__ == "__main__":
    main()
