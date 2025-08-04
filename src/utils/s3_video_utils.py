"""
S3 video input utilities for football analysis project.

This module provides functionality to handle S3 URIs as video input sources,
including authentication, downloading, and streaming from Amazon S3.
"""

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️  boto3 not available. Install with: pip install boto3")

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def is_s3_uri(path: str) -> bool:
    """
    Check if a path is an S3 URI.

    Args:
        path: The path to check

    Returns:
        bool: True if the path is an S3 URI, False otherwise
    """
    if not isinstance(path, str):
        return False
    return path.startswith("s3://")


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into bucket and key components.

    Args:
        s3_uri: S3 URI in format s3://bucket-name/path/to/file

    Returns:
        Tuple[str, str]: (bucket_name, object_key)

    Raises:
        ValueError: If the S3 URI format is invalid
    """
    if not is_s3_uri(s3_uri):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    # Remove s3:// prefix and split
    path = s3_uri[5:]  # Remove 's3://'

    if "/" not in path:
        raise ValueError(f"Invalid S3 URI format - missing object key: {s3_uri}")

    parts = path.split("/", 1)
    bucket_name = parts[0]
    object_key = parts[1]

    if not bucket_name:
        raise ValueError(f"Invalid S3 URI format - empty bucket name: {s3_uri}")

    if not object_key or object_key == "/":
        raise ValueError(f"Invalid S3 URI format - empty object key: {s3_uri}")

    return bucket_name, object_key


def create_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> Optional[boto3.client]:
    """
    Create an S3 client with proper authentication.

    Args:
        aws_access_key_id: AWS access key ID (optional, will use env vars or profile)
        aws_secret_access_key: AWS secret access key (optional, will use env vars or profile)
        region_name: AWS region name (optional, defaults to us-east-1)
        profile_name: AWS profile name (optional, for credential profiles)

    Returns:
        boto3.client or None: S3 client if successful, None otherwise
    """
    if not BOTO3_AVAILABLE:
        print(
            "❌ boto3 is required for S3 functionality. Install with: pip install boto3"
        )
        return None

    # Load environment variables if available
    if DOTENV_AVAILABLE:
        load_dotenv()

    # Use provided credentials or fall back to environment/profile
    session_kwargs = {}

    if profile_name:
        session_kwargs["profile_name"] = profile_name

    if aws_access_key_id and aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key

    try:
        session = boto3.Session(**session_kwargs)
        s3_client = session.client("s3", region_name=region_name or "us-east-1")

        # Test the connection
        s3_client.list_buckets()
        print("✅ Successfully authenticated with AWS S3")
        return s3_client

    except NoCredentialsError:
        print("❌ AWS credentials not found. Please configure your credentials:")
        print(
            "   1. Set environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
        print("   2. Use AWS CLI: aws configure")
        print("   3. Use IAM roles (if running on EC2)")
        print("   4. Provide credentials via command line arguments")
        return None
    except ClientError as e:
        print(f"❌ AWS authentication failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Failed to create S3 client: {e}")
        return None


def download_s3_video(
    s3_uri: str,
    local_path: Optional[str] = None,
    s3_client: Optional[boto3.client] = None,
    **s3_auth_kwargs,
) -> Optional[str]:
    """
    Download a video file from S3 to local storage.

    Args:
        s3_uri: S3 URI of the video file
        local_path: Local path to save the file (optional, will use temp file if None)
        s3_client: Pre-configured S3 client (optional, will create if None)
        **s3_auth_kwargs: Authentication arguments for S3 client creation

    Returns:
        str or None: Local file path if successful, None otherwise
    """
    try:
        bucket_name, object_key = parse_s3_uri(s3_uri)
    except ValueError as e:
        print(f"❌ {e}")
        return None

    # Create S3 client if not provided
    if s3_client is None:
        s3_client = create_s3_client(**s3_auth_kwargs)
        if s3_client is None:
            return None

    # Generate local path if not provided
    if local_path is None:
        # Create a temporary file with the same extension as the S3 object
        file_extension = Path(object_key).suffix or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension, prefix="s3_video_"
        )
        local_path = temp_file.name
        temp_file.close()

    try:
        print(f"📥 Downloading video from S3: {s3_uri}")
        print(f"📁 Local path: {local_path}")

        # Get object info for progress tracking
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
            file_size = response["ContentLength"]
            print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
        except ClientError:
            file_size = None

        start_time = time.time()

        # Download the file
        s3_client.download_file(bucket_name, object_key, local_path)

        download_time = time.time() - start_time
        print(f"✅ Download completed in {download_time:.1f} seconds")

        # Verify the file exists and has content
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"✅ Video file ready: {local_path}")
            return local_path
        else:
            print(f"❌ Downloaded file is empty or missing: {local_path}")
            return None

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            print(f"❌ S3 bucket not found: {bucket_name}")
            print("   Please check:")
            print("   - Bucket name is correct")
            print("   - Bucket exists in the specified region")
            print("   - You have access to the bucket")
        elif error_code == "NoSuchKey":
            print(f"❌ S3 object not found: {object_key}")
            print("   Please check:")
            print("   - Object key (path) is correct")
            print("   - File exists in the bucket")
            print("   - File hasn't been moved or deleted")
        elif error_code == "AccessDenied":
            print(f"❌ Access denied to S3 object: {s3_uri}")
            print("   Please check:")
            print("   - AWS credentials have read permissions")
            print("   - Bucket policy allows access")
            print("   - Object is not encrypted with different key")
        elif error_code == "InvalidAccessKeyId":
            print(f"❌ Invalid AWS access key ID")
            print("   Please check your AWS credentials")
        elif error_code == "SignatureDoesNotMatch":
            print(f"❌ Invalid AWS secret access key")
            print("   Please check your AWS credentials")
        elif error_code == "TokenRefreshRequired":
            print(f"❌ AWS token expired")
            print("   Please refresh your AWS credentials")
        else:
            print(f"❌ S3 error downloading {s3_uri}: {e}")
            print(f"   Error code: {error_code}")
        return None
    except BotoCoreError as e:
        print(f"❌ AWS connection error: {e}")
        print("   Please check:")
        print("   - Internet connection")
        print("   - AWS region is correct")
        print("   - AWS service is available")
        return None
    except Exception as e:
        print(f"❌ Unexpected error downloading {s3_uri}: {e}")
        print("   This might be a network or system issue")
        return None


class S3VideoHandler:
    """
    Handler for S3 video operations with caching and cleanup.
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize S3 video handler.

        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
            profile_name: AWS profile name
            cache_dir: Directory for caching downloaded videos
        """
        self.s3_client = create_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            profile_name=profile_name,
        )

        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.downloaded_files = []  # Track downloaded files for cleanup

    def get_local_path(
        self, s3_uri: str, force_download: bool = False
    ) -> Optional[str]:
        """
        Get local path for S3 video, downloading if necessary.

        Args:
            s3_uri: S3 URI of the video
            force_download: Force re-download even if cached

        Returns:
            str or None: Local file path if successful, None otherwise
        """
        if self.s3_client is None:
            print("❌ S3 client not available")
            return None

        try:
            bucket_name, object_key = parse_s3_uri(s3_uri)
        except ValueError as e:
            print(f"❌ {e}")
            return None

        # Generate cache file path
        safe_key = re.sub(r"[^\w\-_\.]", "_", object_key)
        cache_file = os.path.join(self.cache_dir, f"s3_{bucket_name}_{safe_key}")

        # Check if file exists in cache and is valid
        if (
            not force_download
            and os.path.exists(cache_file)
            and os.path.getsize(cache_file) > 0
        ):
            print(f"📂 Using cached video: {cache_file}")
            return cache_file

        # Download the file
        local_path = download_s3_video(
            s3_uri=s3_uri, local_path=cache_file, s3_client=self.s3_client
        )

        if local_path:
            self.downloaded_files.append(local_path)

        return local_path

    def cleanup(self):
        """Clean up downloaded temporary files."""
        for file_path in self.downloaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"⚠️  Failed to clean up {file_path}: {e}")

        self.downloaded_files.clear()

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()
