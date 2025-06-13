"""
DigitalOcean Spaces (S3-compatible) storage utilities for football analysis project.

This module provides functionality to upload CSV files and output videos to
DigitalOcean Spaces using boto3 S3-compatible API.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️  boto3 not available. Install with: pip install boto3")

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class DigitalOceanSpacesUploader:
    """
    Handles uploading files to DigitalOcean Spaces (S3-compatible storage).
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str = "nyc3",
        endpoint_url: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        """
        Initialize the DigitalOcean Spaces uploader.

        Args:
            access_key_id: DigitalOcean Spaces access key ID
            secret_access_key: DigitalOcean Spaces secret access key
            region: DigitalOcean region (default: nyc3)
            endpoint_url: Custom endpoint URL (auto-generated if None)
            bucket_name: Default bucket name for uploads
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for DigitalOcean Spaces integration. Install with: pip install boto3"
            )

        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.bucket_name = bucket_name

        # Auto-generate endpoint URL if not provided
        if endpoint_url is None:
            self.endpoint_url = f"https://{region}.digitaloceanspaces.com"
        else:
            self.endpoint_url = endpoint_url

        # Initialize boto3 client
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the boto3 S3 client for DigitalOcean Spaces."""
        try:
            self.client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            )
            print(f"✅ Connected to DigitalOcean Spaces in region {self.region}")
        except Exception as e:
            print(f"❌ Failed to initialize DigitalOcean Spaces client: {e}")
            self.client = None

    def test_connection(self, bucket_name: Optional[str] = None) -> bool:
        """
        Test the connection to DigitalOcean Spaces.

        Args:
            bucket_name: Bucket to test (uses default if None)

        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.client:
            return False

        bucket = bucket_name or self.bucket_name
        if not bucket:
            print("❌ No bucket name provided for connection test")
            return False

        try:
            self.client.head_bucket(Bucket=bucket)
            print(f"✅ Successfully connected to bucket '{bucket}'")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                print(f"❌ Bucket '{bucket}' not found")
            elif error_code == "403":
                print(f"❌ Access denied to bucket '{bucket}'. Check permissions.")
            else:
                print(f"❌ Error accessing bucket '{bucket}': {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error testing connection: {e}")
            return False

    def upload_file(
        self,
        local_file_path: str,
        remote_key: str,
        bucket_name: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upload a file to DigitalOcean Spaces.

        Args:
            local_file_path: Path to the local file
            remote_key: Key (path) for the file in the bucket
            bucket_name: Bucket to upload to (uses default if None)
            extra_args: Additional arguments for upload (e.g., metadata, ACL)

        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.client:
            print("❌ DigitalOcean Spaces client not initialized")
            return False

        bucket = bucket_name or self.bucket_name
        if not bucket:
            print("❌ No bucket name provided for upload")
            return False

        if not os.path.exists(local_file_path):
            print(f"❌ Local file not found: {local_file_path}")
            return False

        try:
            file_size = os.path.getsize(local_file_path)
            print(
                f"📤 Uploading {local_file_path} ({file_size / (1024*1024):.1f} MB) to {bucket}/{remote_key}"
            )

            start_time = time.time()

            # Upload the file
            self.client.upload_file(
                local_file_path, bucket, remote_key, ExtraArgs=extra_args or {}
            )

            upload_time = time.time() - start_time
            print(f"✅ Upload completed in {upload_time:.1f} seconds")
            print(f"🔗 File available at: {self.endpoint_url}/{bucket}/{remote_key}")

            return True

        except ClientError as e:
            print(f"❌ Failed to upload {local_file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during upload: {e}")
            return False

    def upload_csv_files(
        self,
        team_csv_path: str,
        player_csv_path: str,
        video_name: str,
        bucket_name: Optional[str] = None,
        folder_prefix: str = "football_analysis",
    ) -> Dict[str, bool]:
        """
        Upload team and player CSV files to DigitalOcean Spaces.

        Args:
            team_csv_path: Path to team statistics CSV
            player_csv_path: Path to player statistics CSV
            video_name: Name of the video (used for organizing files)
            bucket_name: Bucket to upload to (uses default if None)
            folder_prefix: Folder prefix in the bucket

        Returns:
            dict: Upload results for each file
        """
        results = {}

        # Upload team CSV
        team_remote_key = f"{folder_prefix}/csv/{video_name}_team_stats.csv"
        results["team_csv"] = self.upload_file(
            team_csv_path, team_remote_key, bucket_name
        )

        # Upload player CSV
        player_remote_key = f"{folder_prefix}/csv/{video_name}_player_stats.csv"
        results["player_csv"] = self.upload_file(
            player_csv_path, player_remote_key, bucket_name
        )

        return results

    def upload_video(
        self,
        video_path: str,
        video_name: str,
        bucket_name: Optional[str] = None,
        folder_prefix: str = "football_analysis",
    ) -> bool:
        """
        Upload output video to DigitalOcean Spaces.

        Args:
            video_path: Path to the video file
            video_name: Name of the video (used for organizing files)
            bucket_name: Bucket to upload to (uses default if None)
            folder_prefix: Folder prefix in the bucket

        Returns:
            bool: True if upload successful, False otherwise
        """
        video_extension = Path(video_path).suffix
        remote_key = f"{folder_prefix}/videos/{video_name}_output{video_extension}"

        return self.upload_file(video_path, remote_key, bucket_name)


def create_uploader_from_env() -> Optional[DigitalOceanSpacesUploader]:
    """
    Create a DigitalOcean Spaces uploader using environment variables or .env file.

    This function will:
    1. Try to load from .env file if available
    2. Fall back to system environment variables

    Expected environment variables:
    - DO_SPACES_ACCESS_KEY_ID: DigitalOcean Spaces access key ID
    - DO_SPACES_SECRET_ACCESS_KEY: DigitalOcean Spaces secret access key
    - DO_SPACES_REGION: DigitalOcean region (optional, default: nyc3)
    - DO_SPACES_BUCKET: Default bucket name (optional)
    - DO_SPACES_ENDPOINT: Custom endpoint URL (optional)
    - DO_SPACES_FOLDER_PREFIX: Default folder prefix (optional)

    Returns:
        DigitalOceanSpacesUploader or None if credentials not found
    """
    # Try to load from .env file if available
    if DOTENV_AVAILABLE:
        # Look for .env file in current directory and parent directories
        env_file = None
        current_dir = Path.cwd()

        # Check current directory and up to 3 parent directories
        for i in range(4):
            potential_env = current_dir / ".env"
            if potential_env.exists():
                env_file = potential_env
                break
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                break

        if env_file:
            print(f"📄 Loading configuration from {env_file}")
            load_dotenv(env_file)
        else:
            # Try to load from default location
            load_dotenv()

    access_key_id = os.getenv("DO_SPACES_ACCESS_KEY_ID")
    secret_access_key = os.getenv("DO_SPACES_SECRET_ACCESS_KEY")

    if not access_key_id or not secret_access_key:
        print("❌ DigitalOcean Spaces credentials not found")
        print("   Options:")
        print("   1. Create a .env file with your credentials (see .env.example)")
        print(
            "   2. Set environment variables: DO_SPACES_ACCESS_KEY_ID and DO_SPACES_SECRET_ACCESS_KEY"
        )
        print(
            "   3. Use command line arguments: --spaces-access-key-id and --spaces-secret-access-key"
        )
        return None

    region = os.getenv("DO_SPACES_REGION", "nyc3")
    bucket_name = os.getenv("DO_SPACES_BUCKET")
    endpoint_url = os.getenv("DO_SPACES_ENDPOINT")

    try:
        uploader = DigitalOceanSpacesUploader(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            endpoint_url=endpoint_url,
            bucket_name=bucket_name,
        )
        return uploader
    except Exception as e:
        print(f"❌ Failed to create DigitalOcean Spaces uploader: {e}")
        return None


def create_uploader_from_args(
    access_key_id: str,
    secret_access_key: str,
    region: str = "nyc3",
    bucket_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> Optional[DigitalOceanSpacesUploader]:
    """
    Create a DigitalOcean Spaces uploader using provided arguments.

    Args:
        access_key_id: DigitalOcean Spaces access key ID
        secret_access_key: DigitalOcean Spaces secret access key
        region: DigitalOcean region (default: nyc3)
        bucket_name: Default bucket name (optional)
        endpoint_url: Custom endpoint URL (optional)

    Returns:
        DigitalOceanSpacesUploader or None if creation failed
    """
    try:
        uploader = DigitalOceanSpacesUploader(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region,
            endpoint_url=endpoint_url,
            bucket_name=bucket_name,
        )
        return uploader
    except Exception as e:
        print(f"❌ Failed to create DigitalOcean Spaces uploader: {e}")
        return None
