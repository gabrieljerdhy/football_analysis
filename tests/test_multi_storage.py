#!/usr/bin/env python3
"""
Tests for multi-provider object storage functionality.

This module contains comprehensive tests for the multi-storage system
including URI detection, parsing, and provider-specific functionality.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.multi_storage_utils import (
    StorageProvider,
    detect_storage_provider,
    is_object_storage_uri,
    parse_storage_uri,
    MultiStorageHandler,
)


class TestStorageProviderDetection(unittest.TestCase):
    """Test storage provider detection and URI validation."""
    
    def test_aws_s3_detection(self):
        """Test AWS S3 URI detection."""
        test_cases = [
            ("s3://bucket/file.mp4", StorageProvider.AWS_S3, True),
            ("s3://my-bucket/path/to/video.avi", StorageProvider.AWS_S3, True),
            ("S3://BUCKET/FILE.MP4", StorageProvider.AWS_S3, True),  # Case insensitive
        ]
        
        for uri, expected_provider, expected_valid in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertEqual(is_valid, expected_valid)
    
    def test_digitalocean_spaces_detection(self):
        """Test DigitalOcean Spaces URI detection."""
        test_cases = [
            ("spaces://bucket.nyc3.digitaloceanspaces.com/video.mp4", StorageProvider.DIGITALOCEAN_SPACES, True),
            ("https://bucket.fra1.digitaloceanspaces.com/path/video.avi", StorageProvider.DIGITALOCEAN_SPACES, True),
        ]
        
        for uri, expected_provider, expected_valid in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertEqual(is_valid, expected_valid)
    
    def test_google_cloud_detection(self):
        """Test Google Cloud Storage URI detection."""
        test_cases = [
            ("gs://bucket/video.mp4", StorageProvider.GOOGLE_CLOUD, True),
            ("gs://my-bucket/path/to/video.avi", StorageProvider.GOOGLE_CLOUD, True),
            ("GS://BUCKET/FILE.MP4", StorageProvider.GOOGLE_CLOUD, True),  # Case insensitive
        ]
        
        for uri, expected_provider, expected_valid in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertEqual(is_valid, expected_valid)
    
    def test_azure_blob_detection(self):
        """Test Azure Blob Storage URI detection."""
        test_cases = [
            ("azure://account.blob.core.windows.net/container/video.mp4", StorageProvider.AZURE_BLOB, True),
            ("https://account.blob.core.windows.net/container/path/video.avi", StorageProvider.AZURE_BLOB, True),
        ]
        
        for uri, expected_provider, expected_valid in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertEqual(is_valid, expected_valid)
    
    def test_minio_detection(self):
        """Test MinIO URI detection."""
        test_cases = [
            ("minio://localhost:9000/bucket/video.mp4", StorageProvider.MINIO, True),
            ("minio://minio.example.com/bucket/path/video.avi", StorageProvider.MINIO, True),
        ]
        
        for uri, expected_provider, expected_valid in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertEqual(is_valid, expected_valid)
    
    def test_invalid_uris(self):
        """Test invalid URI detection."""
        invalid_uris = [
            "http://example.com/video.mp4",
            "/local/path/video.mp4",
            "ftp://server/video.mp4",
            "",
            None,
            123,
        ]
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertFalse(is_valid)
    
    def test_is_object_storage_uri(self):
        """Test the convenience function for checking object storage URIs."""
        valid_uris = [
            "s3://bucket/video.mp4",
            "gs://bucket/video.mp4",
            "azure://account.blob.core.windows.net/container/video.mp4",
            "spaces://bucket.nyc3.digitaloceanspaces.com/video.mp4",
            "minio://localhost:9000/bucket/video.mp4",
        ]
        
        invalid_uris = [
            "http://example.com/video.mp4",
            "/local/path/video.mp4",
            "",
        ]
        
        for uri in valid_uris:
            with self.subTest(uri=uri):
                self.assertTrue(is_object_storage_uri(uri))
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                self.assertFalse(is_object_storage_uri(uri))


class TestURIParsing(unittest.TestCase):
    """Test URI parsing for different storage providers."""
    
    def test_s3_uri_parsing(self):
        """Test AWS S3 URI parsing."""
        test_cases = [
            ("s3://bucket/file.mp4", {
                'provider': StorageProvider.AWS_S3,
                'bucket': 'bucket',
                'key': 'file.mp4',
                'endpoint': None,
                'region': None
            }),
            ("s3://my-bucket/path/to/video.avi", {
                'provider': StorageProvider.AWS_S3,
                'bucket': 'my-bucket',
                'key': 'path/to/video.avi',
                'endpoint': None,
                'region': None
            }),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = parse_storage_uri(uri)
                self.assertEqual(result, expected)
    
    def test_spaces_uri_parsing(self):
        """Test DigitalOcean Spaces URI parsing."""
        test_cases = [
            ("spaces://bucket.nyc3.digitaloceanspaces.com/video.mp4", {
                'provider': StorageProvider.DIGITALOCEAN_SPACES,
                'bucket': 'bucket',
                'key': 'video.mp4',
                'endpoint': 'https://nyc3.digitaloceanspaces.com',
                'region': 'nyc3'
            }),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = parse_storage_uri(uri)
                self.assertEqual(result, expected)
    
    def test_gcs_uri_parsing(self):
        """Test Google Cloud Storage URI parsing."""
        test_cases = [
            ("gs://bucket/video.mp4", {
                'provider': StorageProvider.GOOGLE_CLOUD,
                'bucket': 'bucket',
                'key': 'video.mp4',
                'endpoint': None,
                'region': None
            }),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = parse_storage_uri(uri)
                self.assertEqual(result, expected)
    
    def test_azure_uri_parsing(self):
        """Test Azure Blob Storage URI parsing."""
        test_cases = [
            ("azure://account.blob.core.windows.net/container/video.mp4", {
                'provider': StorageProvider.AZURE_BLOB,
                'account': 'account',
                'container': 'container',
                'key': 'video.mp4',
                'endpoint': 'https://account.blob.core.windows.net'
            }),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = parse_storage_uri(uri)
                self.assertEqual(result, expected)
    
    def test_minio_uri_parsing(self):
        """Test MinIO URI parsing."""
        test_cases = [
            ("minio://localhost:9000/bucket/video.mp4", {
                'provider': StorageProvider.MINIO,
                'bucket': 'bucket',
                'key': 'video.mp4',
                'endpoint': 'http://localhost:9000',
                'region': None
            }),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                result = parse_storage_uri(uri)
                self.assertEqual(result, expected)
    
    def test_invalid_uri_parsing(self):
        """Test parsing of invalid URIs."""
        invalid_uris = [
            "http://example.com/video.mp4",
            "s3://bucket-only",
            "gs://",
            "",
        ]
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                with self.assertRaises(ValueError):
                    parse_storage_uri(uri)


class TestMultiStorageHandler(unittest.TestCase):
    """Test the unified multi-storage handler."""
    
    def test_handler_initialization(self):
        """Test MultiStorageHandler initialization."""
        config = {
            'aws_access_key_id': 'test_key',
            'aws_secret_access_key': 'test_secret',
        }
        
        handler = MultiStorageHandler(**config)
        self.assertEqual(handler.config, config)
        self.assertEqual(len(handler.clients), 0)
        self.assertEqual(len(handler.downloaded_files), 0)
    
    def test_provider_detection_integration(self):
        """Test that the handler correctly detects storage providers."""
        handler = MultiStorageHandler()
        
        test_cases = [
            ("s3://bucket/video.mp4", StorageProvider.AWS_S3),
            ("gs://bucket/video.mp4", StorageProvider.GOOGLE_CLOUD),
            ("azure://account.blob.core.windows.net/container/video.mp4", StorageProvider.AZURE_BLOB),
            ("spaces://bucket.nyc3.digitaloceanspaces.com/video.mp4", StorageProvider.DIGITALOCEAN_SPACES),
            ("minio://localhost:9000/bucket/video.mp4", StorageProvider.MINIO),
        ]
        
        for uri, expected_provider in test_cases:
            with self.subTest(uri=uri):
                provider, is_valid = detect_storage_provider(uri)
                self.assertEqual(provider, expected_provider)
                self.assertTrue(is_valid)
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        handler = MultiStorageHandler()
        
        # Mock some downloaded files
        handler.downloaded_files = ["/tmp/file1.mp4", "/tmp/file2.mp4"]
        
        with patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            handler.cleanup()
            
            # Verify files were removed
            self.assertEqual(mock_remove.call_count, 2)
            mock_remove.assert_any_call("/tmp/file1.mp4")
            mock_remove.assert_any_call("/tmp/file2.mp4")
            
            # Verify list was cleared
            self.assertEqual(len(handler.downloaded_files), 0)


def run_tests():
    """Run all multi-storage tests."""
    print("🧪 Running Multi-Storage Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStorageProviderDetection,
        TestURIParsing,
        TestMultiStorageHandler,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All multi-storage tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
