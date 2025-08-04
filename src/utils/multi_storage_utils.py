"""
Multi-Provider Object Storage Utilities for Football Analysis Project.

This module provides unified support for multiple object storage providers including:
- AWS S3
- DigitalOcean Spaces
- Google Cloud Storage
- Azure Blob Storage
- MinIO and other S3-compatible services

All providers are accessed through a unified interface while maintaining
provider-specific authentication and configuration options.
"""

import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage as gcs

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from azure.core.exceptions import AzureError
    from azure.storage.blob import BlobServiceClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class StorageProvider:
    """Enumeration of supported storage providers."""

    AWS_S3 = "aws_s3"
    DIGITALOCEAN_SPACES = "digitalocean_spaces"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"
    MINIO = "minio"
    S3_COMPATIBLE = "s3_compatible"


def detect_storage_provider(uri: str) -> Tuple[str, bool]:
    """
    Detect the storage provider from URI format.

    Args:
        uri: Storage URI to analyze

    Returns:
        Tuple[str, bool]: (provider_type, is_valid)
    """
    if not isinstance(uri, str):
        return StorageProvider.AWS_S3, False

    uri_lower = uri.lower()

    # AWS S3
    if uri_lower.startswith("s3://"):
        return StorageProvider.AWS_S3, True

    # DigitalOcean Spaces
    if uri_lower.startswith("spaces://") or "digitaloceanspaces.com" in uri_lower:
        return StorageProvider.DIGITALOCEAN_SPACES, True

    # Google Cloud Storage
    if uri_lower.startswith("gs://"):
        return StorageProvider.GOOGLE_CLOUD, True

    # Azure Blob Storage
    if uri_lower.startswith("azure://") or "blob.core.windows.net" in uri_lower:
        return StorageProvider.AZURE_BLOB, True

    # MinIO (often uses custom domains)
    if uri_lower.startswith("minio://"):
        return StorageProvider.MINIO, True

    # Check for S3-compatible with custom endpoint
    if "://" in uri and any(
        keyword in uri_lower for keyword in ["s3", "storage", "object"]
    ):
        return StorageProvider.S3_COMPATIBLE, True

    return StorageProvider.AWS_S3, False


def is_object_storage_uri(uri: str) -> bool:
    """
    Check if a URI is from any supported object storage provider.

    Args:
        uri: URI to check

    Returns:
        bool: True if URI is from a supported storage provider
    """
    _, is_valid = detect_storage_provider(uri)
    return is_valid


def parse_storage_uri(uri: str) -> Dict[str, str]:
    """
    Parse a storage URI into its components.

    Args:
        uri: Storage URI to parse

    Returns:
        Dict containing parsed components

    Raises:
        ValueError: If URI format is invalid
    """
    provider, is_valid = detect_storage_provider(uri)

    if not is_valid:
        raise ValueError(f"Invalid or unsupported storage URI format: {uri}")

    if provider == StorageProvider.AWS_S3:
        return _parse_s3_uri(uri)
    elif provider == StorageProvider.DIGITALOCEAN_SPACES:
        return _parse_spaces_uri(uri)
    elif provider == StorageProvider.GOOGLE_CLOUD:
        return _parse_gcs_uri(uri)
    elif provider == StorageProvider.AZURE_BLOB:
        return _parse_azure_uri(uri)
    elif provider == StorageProvider.MINIO:
        return _parse_minio_uri(uri)
    elif provider == StorageProvider.S3_COMPATIBLE:
        return _parse_s3_compatible_uri(uri)
    else:
        raise ValueError(f"Unsupported storage provider: {provider}")


def _parse_s3_uri(uri: str) -> Dict[str, str]:
    """Parse AWS S3 URI: s3://bucket/path/to/file"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {uri}")

    path = uri[5:]  # Remove 's3://'
    if "/" not in path:
        raise ValueError(f"Invalid S3 URI - missing object key: {uri}")

    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1]

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI - empty bucket or key: {uri}")

    return {
        "provider": StorageProvider.AWS_S3,
        "bucket": bucket,
        "key": key,
        "endpoint": None,
        "region": None,
    }


def _parse_spaces_uri(uri: str) -> Dict[str, str]:
    """Parse DigitalOcean Spaces URI: spaces://bucket.region.digitaloceanspaces.com/path"""
    if uri.startswith("spaces://"):
        # Format: spaces://bucket.region.digitaloceanspaces.com/path
        path = uri[9:]  # Remove 'spaces://'
        if "/" not in path:
            raise ValueError(f"Invalid Spaces URI - missing object key: {uri}")

        domain_part, key = path.split("/", 1)

        # Extract bucket and region from domain
        if ".digitaloceanspaces.com" not in domain_part:
            raise ValueError(
                f"Invalid Spaces URI - missing digitaloceanspaces.com: {uri}"
            )

        bucket_region = domain_part.replace(".digitaloceanspaces.com", "")
        if "." not in bucket_region:
            raise ValueError(
                f"Invalid Spaces URI - cannot determine bucket and region: {uri}"
            )

        bucket, region = bucket_region.split(".", 1)
        endpoint = f"https://{region}.digitaloceanspaces.com"

    else:
        # Alternative format with full URL
        parsed = urlparse(uri)
        if "digitaloceanspaces.com" not in parsed.netloc:
            raise ValueError(f"Invalid Spaces URI: {uri}")

        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"Invalid Spaces URI - missing object key: {uri}")

        # Extract bucket and region from netloc
        parts = parsed.netloc.split(".")
        if len(parts) < 3:
            raise ValueError(f"Invalid Spaces URI format: {uri}")

        bucket = parts[0]
        region = parts[1]
        endpoint = f"https://{region}.digitaloceanspaces.com"

    return {
        "provider": StorageProvider.DIGITALOCEAN_SPACES,
        "bucket": bucket,
        "key": key,
        "endpoint": endpoint,
        "region": region,
    }


def _parse_gcs_uri(uri: str) -> Dict[str, str]:
    """Parse Google Cloud Storage URI: gs://bucket/path/to/file"""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI format: {uri}")

    path = uri[5:]  # Remove 'gs://'
    if "/" not in path:
        raise ValueError(f"Invalid GCS URI - missing object key: {uri}")

    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1]

    if not bucket or not key:
        raise ValueError(f"Invalid GCS URI - empty bucket or key: {uri}")

    return {
        "provider": StorageProvider.GOOGLE_CLOUD,
        "bucket": bucket,
        "key": key,
        "endpoint": None,
        "region": None,
    }


def _parse_azure_uri(uri: str) -> Dict[str, str]:
    """Parse Azure Blob Storage URI: azure://account.blob.core.windows.net/container/path"""
    if uri.startswith("azure://"):
        path = uri[8:]  # Remove 'azure://'
    else:
        parsed = urlparse(uri)
        if "blob.core.windows.net" not in parsed.netloc:
            raise ValueError(f"Invalid Azure URI: {uri}")
        path = parsed.netloc + parsed.path

    if "/" not in path:
        raise ValueError(f"Invalid Azure URI - missing blob path: {uri}")

    # Extract account, container, and blob path
    if ".blob.core.windows.net/" in path:
        account_part, blob_part = path.split(".blob.core.windows.net/", 1)
        account = account_part

        if "/" not in blob_part:
            raise ValueError(f"Invalid Azure URI - missing blob name: {uri}")

        container, key = blob_part.split("/", 1)
    else:
        raise ValueError(f"Invalid Azure URI format: {uri}")

    return {
        "provider": StorageProvider.AZURE_BLOB,
        "account": account,
        "container": container,
        "key": key,
        "endpoint": f"https://{account}.blob.core.windows.net",
    }


def _parse_minio_uri(uri: str) -> Dict[str, str]:
    """Parse MinIO URI: minio://endpoint/bucket/path/to/file"""
    if not uri.startswith("minio://"):
        raise ValueError(f"Invalid MinIO URI format: {uri}")

    path = uri[8:]  # Remove 'minio://'
    if "/" not in path:
        raise ValueError(f"Invalid MinIO URI - missing bucket/key: {uri}")

    parts = path.split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid MinIO URI - missing bucket or key: {uri}")

    endpoint = parts[0]
    bucket = parts[1]
    key = parts[2]

    return {
        "provider": StorageProvider.MINIO,
        "bucket": bucket,
        "key": key,
        "endpoint": f"http://{endpoint}",  # MinIO often uses HTTP
        "region": None,
    }


def _parse_s3_compatible_uri(uri: str) -> Dict[str, str]:
    """Parse generic S3-compatible URI"""
    parsed = urlparse(uri)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid S3-compatible URI format: {uri}")

    path = parsed.path.lstrip("/")
    if "/" not in path:
        raise ValueError(f"Invalid S3-compatible URI - missing object key: {uri}")

    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1]

    endpoint = f"{parsed.scheme}://{parsed.netloc}"

    return {
        "provider": StorageProvider.S3_COMPATIBLE,
        "bucket": bucket,
        "key": key,
        "endpoint": endpoint,
        "region": None,
    }


class StorageClient(ABC):
    """Abstract base class for storage provider clients."""

    def __init__(self, **kwargs):
        self.provider = None
        self.client = None
        self.config = kwargs

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the storage provider."""
        pass

    @abstractmethod
    def download_file(self, uri: str, local_path: str) -> bool:
        """Download a file from storage to local path."""
        pass

    @abstractmethod
    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get file information (size, metadata, etc.)."""
        pass

    def test_connection(self) -> bool:
        """Test connection to storage provider."""
        try:
            return self.authenticate()
        except Exception:
            return False


class S3Client(StorageClient):
    """AWS S3 storage client."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = StorageProvider.AWS_S3

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 support. Install with: pip install boto3"
            )

    def authenticate(self) -> bool:
        """Authenticate with AWS S3."""
        try:
            session_kwargs = {}

            # Use provided credentials or fall back to environment/profile
            if self.config.get("aws_access_key_id") and self.config.get(
                "aws_secret_access_key"
            ):
                session_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
                session_kwargs["aws_secret_access_key"] = self.config[
                    "aws_secret_access_key"
                ]

            if self.config.get("aws_profile"):
                session_kwargs["profile_name"] = self.config["aws_profile"]

            session = boto3.Session(**session_kwargs)
            self.client = session.client(
                "s3", region_name=self.config.get("aws_region", "us-east-1")
            )

            # Test authentication
            self.client.list_buckets()
            return True

        except Exception as e:
            print(f"❌ AWS S3 authentication failed: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            print(f"📥 Downloading from S3: {uri}")
            start_time = time.time()

            self.client.download_file(bucket, key, local_path)

            download_time = time.time() - start_time
            print(f"✅ S3 download completed in {download_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ S3 download failed: {e}")
            return False

    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get S3 object information."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            response = self.client.head_object(Bucket=bucket, Key=key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"],
            }
        except Exception:
            return None


class SpacesClient(StorageClient):
    """DigitalOcean Spaces storage client."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = StorageProvider.DIGITALOCEAN_SPACES

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Spaces support. Install with: pip install boto3"
            )

    def authenticate(self) -> bool:
        """Authenticate with DigitalOcean Spaces."""
        try:
            access_key = self.config.get("spaces_access_key_id") or os.getenv(
                "DO_SPACES_ACCESS_KEY_ID"
            )
            secret_key = self.config.get("spaces_secret_access_key") or os.getenv(
                "DO_SPACES_SECRET_ACCESS_KEY"
            )

            if not access_key or not secret_key:
                print("❌ DigitalOcean Spaces credentials not found")
                return False

            # Spaces uses S3-compatible API
            self.client = boto3.client(
                "s3",
                endpoint_url=self.config.get(
                    "endpoint", "https://nyc3.digitaloceanspaces.com"
                ),
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.config.get("region", "nyc3"),
            )

            # Test authentication
            self.client.list_buckets()
            return True

        except Exception as e:
            print(f"❌ DigitalOcean Spaces authentication failed: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from Spaces."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            print(f"📥 Downloading from DigitalOcean Spaces: {uri}")
            start_time = time.time()

            self.client.download_file(bucket, key, local_path)

            download_time = time.time() - start_time
            print(f"✅ Spaces download completed in {download_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ Spaces download failed: {e}")
            return False

    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get Spaces object information."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            response = self.client.head_object(Bucket=bucket, Key=key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"],
            }
        except Exception:
            return None


class GCSClient(StorageClient):
    """Google Cloud Storage client."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = StorageProvider.GOOGLE_CLOUD

        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for GCS support. Install with: pip install google-cloud-storage"
            )

    def authenticate(self) -> bool:
        """Authenticate with Google Cloud Storage."""
        try:
            # Use service account key file if provided
            if self.config.get("gcs_service_account_path"):
                self.client = gcs.Client.from_service_account_json(
                    self.config["gcs_service_account_path"]
                )
            else:
                # Use application default credentials
                self.client = gcs.Client()

            # Test authentication by listing buckets
            list(self.client.list_buckets(max_results=1))
            return True

        except Exception as e:
            print(f"❌ Google Cloud Storage authentication failed: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from GCS."""
        try:
            parsed = parse_storage_uri(uri)
            bucket_name = parsed["bucket"]
            blob_name = parsed["key"]

            print(f"📥 Downloading from Google Cloud Storage: {uri}")
            start_time = time.time()

            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            download_time = time.time() - start_time
            print(f"✅ GCS download completed in {download_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ GCS download failed: {e}")
            return False

    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get GCS object information."""
        try:
            parsed = parse_storage_uri(uri)
            bucket_name = parsed["bucket"]
            blob_name = parsed["key"]

            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.reload()

            return {"size": blob.size, "last_modified": blob.updated, "etag": blob.etag}
        except Exception:
            return None


class AzureBlobClient(StorageClient):
    """Azure Blob Storage client."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = StorageProvider.AZURE_BLOB

        if not AZURE_AVAILABLE:
            raise ImportError(
                "azure-storage-blob is required for Azure support. Install with: pip install azure-storage-blob"
            )

    def authenticate(self) -> bool:
        """Authenticate with Azure Blob Storage."""
        try:
            account_name = self.config.get("azure_account_name") or os.getenv(
                "AZURE_STORAGE_ACCOUNT"
            )
            account_key = self.config.get("azure_account_key") or os.getenv(
                "AZURE_STORAGE_KEY"
            )
            sas_token = self.config.get("azure_sas_token") or os.getenv(
                "AZURE_STORAGE_SAS_TOKEN"
            )

            if account_name and account_key:
                account_url = f"https://{account_name}.blob.core.windows.net"
                self.client = BlobServiceClient(
                    account_url=account_url, credential=account_key
                )
            elif account_name and sas_token:
                account_url = f"https://{account_name}.blob.core.windows.net"
                self.client = BlobServiceClient(
                    account_url=account_url, credential=sas_token
                )
            else:
                print("❌ Azure credentials not found")
                return False

            # Test authentication
            list(self.client.list_containers(max_results=1))
            return True

        except Exception as e:
            print(f"❌ Azure Blob Storage authentication failed: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            parsed = parse_storage_uri(uri)
            container_name = parsed["container"]
            blob_name = parsed["key"]

            print(f"📥 Downloading from Azure Blob Storage: {uri}")
            start_time = time.time()

            blob_client = self.client.get_blob_client(
                container=container_name, blob=blob_name
            )

            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            download_time = time.time() - start_time
            print(f"✅ Azure download completed in {download_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ Azure download failed: {e}")
            return False

    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get Azure blob information."""
        try:
            parsed = parse_storage_uri(uri)
            container_name = parsed["container"]
            blob_name = parsed["key"]

            blob_client = self.client.get_blob_client(
                container=container_name, blob=blob_name
            )
            properties = blob_client.get_blob_properties()

            return {
                "size": properties.size,
                "last_modified": properties.last_modified,
                "etag": properties.etag,
            }
        except Exception:
            return None


class MinIOClient(StorageClient):
    """MinIO storage client (S3-compatible)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = StorageProvider.MINIO

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for MinIO support. Install with: pip install boto3"
            )

    def authenticate(self) -> bool:
        """Authenticate with MinIO."""
        try:
            access_key = self.config.get("minio_access_key") or os.getenv(
                "MINIO_ACCESS_KEY"
            )
            secret_key = self.config.get("minio_secret_key") or os.getenv(
                "MINIO_SECRET_KEY"
            )
            endpoint = self.config.get("minio_endpoint") or os.getenv("MINIO_ENDPOINT")

            if not access_key or not secret_key or not endpoint:
                print("❌ MinIO credentials or endpoint not found")
                return False

            # MinIO uses S3-compatible API
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",  # MinIO doesn't require specific region
            )

            # Test authentication
            self.client.list_buckets()
            return True

        except Exception as e:
            print(f"❌ MinIO authentication failed: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from MinIO."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            print(f"📥 Downloading from MinIO: {uri}")
            start_time = time.time()

            self.client.download_file(bucket, key, local_path)

            download_time = time.time() - start_time
            print(f"✅ MinIO download completed in {download_time:.1f} seconds")
            return True

        except Exception as e:
            print(f"❌ MinIO download failed: {e}")
            return False

    def get_file_info(self, uri: str) -> Optional[Dict]:
        """Get MinIO object information."""
        try:
            parsed = parse_storage_uri(uri)
            bucket = parsed["bucket"]
            key = parsed["key"]

            response = self.client.head_object(Bucket=bucket, Key=key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"],
            }
        except Exception:
            return None


class MultiStorageHandler:
    """
    Unified handler for multiple object storage providers.

    This class provides a single interface for downloading videos from
    various object storage providers including AWS S3, DigitalOcean Spaces,
    Google Cloud Storage, Azure Blob Storage, and MinIO.
    """

    def __init__(self, **config):
        """
        Initialize multi-storage handler.

        Args:
            **config: Configuration parameters for various storage providers
        """
        self.config = config
        self.clients = {}
        self.downloaded_files = []

        # Load environment variables if available
        if DOTENV_AVAILABLE:
            load_dotenv()

    def _get_client(self, provider: str) -> Optional[StorageClient]:
        """Get or create a client for the specified provider."""
        if provider in self.clients:
            return self.clients[provider]

        try:
            if provider == StorageProvider.AWS_S3:
                client = S3Client(**self.config)
            elif provider == StorageProvider.DIGITALOCEAN_SPACES:
                client = SpacesClient(**self.config)
            elif provider == StorageProvider.GOOGLE_CLOUD:
                client = GCSClient(**self.config)
            elif provider == StorageProvider.AZURE_BLOB:
                client = AzureBlobClient(**self.config)
            elif provider == StorageProvider.MINIO:
                client = MinIOClient(**self.config)
            elif provider == StorageProvider.S3_COMPATIBLE:
                # Use S3Client with custom endpoint
                client = S3Client(**self.config)
            else:
                print(f"❌ Unsupported storage provider: {provider}")
                return None

            if client.authenticate():
                self.clients[provider] = client
                return client
            else:
                return None

        except Exception as e:
            print(f"❌ Failed to create {provider} client: {e}")
            return None

    def download_video(
        self, uri: str, local_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Download a video from any supported object storage provider.

        Args:
            uri: Storage URI (s3://, gs://, azure://, etc.)
            local_path: Local path to save file (optional, will use temp file if None)

        Returns:
            str or None: Local file path if successful, None otherwise
        """
        try:
            # Detect storage provider
            provider, is_valid = detect_storage_provider(uri)
            if not is_valid:
                print(f"❌ Unsupported storage URI: {uri}")
                return None

            print(f"🌐 Detected {provider} storage: {uri}")

            # Get client for provider
            client = self._get_client(provider)
            if not client:
                print(f"❌ Failed to authenticate with {provider}")
                return None

            # Generate local path if not provided
            if local_path is None:
                parsed = parse_storage_uri(uri)
                file_extension = Path(parsed.get("key", "video")).suffix or ".mp4"
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension, prefix=f"{provider}_video_"
                )
                local_path = temp_file.name
                temp_file.close()

            # Download file
            if client.download_file(uri, local_path):
                # Verify file exists and has content
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    self.downloaded_files.append(local_path)
                    print(f"✅ Video ready: {local_path}")
                    return local_path
                else:
                    print(f"❌ Downloaded file is empty or missing: {local_path}")
                    return None
            else:
                return None

        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            return None

    def get_video_info(self, uri: str) -> Optional[Dict]:
        """Get video file information from storage."""
        try:
            provider, is_valid = detect_storage_provider(uri)
            if not is_valid:
                return None

            client = self._get_client(provider)
            if not client:
                return None

            return client.get_file_info(uri)

        except Exception:
            return None

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
