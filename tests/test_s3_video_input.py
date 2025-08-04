#!/usr/bin/env python3
"""
Tests for S3 video input functionality.

This module contains unit tests and integration tests for S3 video input features
including URI parsing, authentication, and video downloading.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.s3_video_utils import (
    is_s3_uri,
    parse_s3_uri,
    create_s3_client,
    download_s3_video,
    S3VideoHandler,
)


class TestS3URIValidation(unittest.TestCase):
    """Test S3 URI validation and parsing functions."""
    
    def test_is_s3_uri_valid_uris(self):
        """Test valid S3 URIs."""
        valid_uris = [
            "s3://bucket/file.mp4",
            "s3://my-bucket/path/to/video.avi",
            "s3://bucket-name/folder/subfolder/movie.mov",
            "s3://test123/video.mkv",
        ]
        
        for uri in valid_uris:
            with self.subTest(uri=uri):
                self.assertTrue(is_s3_uri(uri))
    
    def test_is_s3_uri_invalid_uris(self):
        """Test invalid S3 URIs."""
        invalid_uris = [
            "http://example.com/video.mp4",
            "https://bucket.s3.amazonaws.com/video.mp4",
            "/local/path/video.mp4",
            "file://video.mp4",
            "ftp://server/video.mp4",
            "",
            None,
            123,
        ]
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                self.assertFalse(is_s3_uri(uri))
    
    def test_parse_s3_uri_valid(self):
        """Test parsing valid S3 URIs."""
        test_cases = [
            ("s3://bucket/file.mp4", ("bucket", "file.mp4")),
            ("s3://my-bucket/path/to/video.avi", ("my-bucket", "path/to/video.avi")),
            ("s3://test-123/folder/subfolder/movie.mov", ("test-123", "folder/subfolder/movie.mov")),
        ]
        
        for uri, expected in test_cases:
            with self.subTest(uri=uri):
                bucket, key = parse_s3_uri(uri)
                self.assertEqual((bucket, key), expected)
    
    def test_parse_s3_uri_invalid(self):
        """Test parsing invalid S3 URIs."""
        invalid_uris = [
            "http://example.com/video.mp4",
            "s3://",
            "s3://bucket",
            "s3://bucket/",
            "s3:///file.mp4",
            "",
        ]
        
        for uri in invalid_uris:
            with self.subTest(uri=uri):
                with self.assertRaises(ValueError):
                    parse_s3_uri(uri)


class TestS3ClientCreation(unittest.TestCase):
    """Test S3 client creation and authentication."""
    
    @patch('src.utils.s3_video_utils.boto3')
    def test_create_s3_client_with_credentials(self, mock_boto3):
        """Test creating S3 client with explicit credentials."""
        # Mock boto3 session and client
        mock_session = Mock()
        mock_client = Mock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        
        # Test client creation
        client = create_s3_client(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2"
        )
        
        # Verify session was created with correct parameters
        mock_boto3.Session.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        # Verify client was created with correct region
        mock_session.client.assert_called_once_with('s3', region_name='us-west-2')
        
        self.assertEqual(client, mock_client)
    
    @patch('src.utils.s3_video_utils.boto3')
    def test_create_s3_client_with_profile(self, mock_boto3):
        """Test creating S3 client with AWS profile."""
        mock_session = Mock()
        mock_client = Mock()
        mock_boto3.Session.return_value = mock_session
        mock_session.client.return_value = mock_client
        
        client = create_s3_client(profile_name="test-profile")
        
        mock_boto3.Session.assert_called_once_with(profile_name="test-profile")
        self.assertEqual(client, mock_client)
    
    @patch('src.utils.s3_video_utils.BOTO3_AVAILABLE', False)
    def test_create_s3_client_no_boto3(self):
        """Test client creation when boto3 is not available."""
        client = create_s3_client()
        self.assertIsNone(client)


class TestS3VideoDownload(unittest.TestCase):
    """Test S3 video downloading functionality."""
    
    @patch('src.utils.s3_video_utils.create_s3_client')
    def test_download_s3_video_success(self, mock_create_client):
        """Test successful S3 video download."""
        # Mock S3 client
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {'ContentLength': 1024000}
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Mock download_file to create a file
            def mock_download_file(bucket, key, local_path):
                with open(local_path, 'wb') as f:
                    f.write(b'test video content')
            
            mock_client.download_file.side_effect = mock_download_file
            
            # Test download
            result = download_s3_video(
                "s3://test-bucket/video.mp4",
                local_path=temp_path,
                s3_client=mock_client
            )
            
            # Verify result
            self.assertEqual(result, temp_path)
            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)
            
            # Verify S3 calls
            mock_client.head_object.assert_called_once_with(
                Bucket='test-bucket', Key='video.mp4'
            )
            mock_client.download_file.assert_called_once_with(
                'test-bucket', 'video.mp4', temp_path
            )
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_download_s3_video_invalid_uri(self):
        """Test download with invalid S3 URI."""
        result = download_s3_video("invalid-uri")
        self.assertIsNone(result)
    
    @patch('src.utils.s3_video_utils.create_s3_client')
    def test_download_s3_video_no_client(self, mock_create_client):
        """Test download when S3 client creation fails."""
        mock_create_client.return_value = None
        
        result = download_s3_video("s3://bucket/video.mp4")
        self.assertIsNone(result)


class TestS3VideoHandler(unittest.TestCase):
    """Test S3VideoHandler class functionality."""
    
    @patch('src.utils.s3_video_utils.create_s3_client')
    def test_s3_video_handler_initialization(self, mock_create_client):
        """Test S3VideoHandler initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        handler = S3VideoHandler(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-east-1"
        )
        
        self.assertEqual(handler.s3_client, mock_client)
        mock_create_client.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-east-1",
            profile_name=None
        )
    
    @patch('src.utils.s3_video_utils.download_s3_video')
    def test_get_local_path_success(self, mock_download):
        """Test getting local path for S3 video."""
        # Mock successful download
        mock_download.return_value = "/tmp/cached_video.mp4"
        
        handler = S3VideoHandler()
        handler.s3_client = Mock()  # Mock client
        
        result = handler.get_local_path("s3://bucket/video.mp4")
        
        self.assertEqual(result, "/tmp/cached_video.mp4")
        self.assertIn("/tmp/cached_video.mp4", handler.downloaded_files)
    
    def test_get_local_path_no_client(self):
        """Test getting local path when S3 client is not available."""
        handler = S3VideoHandler()
        handler.s3_client = None
        
        result = handler.get_local_path("s3://bucket/video.mp4")
        self.assertIsNone(result)
    
    @patch('os.path.exists')
    @patch('os.remove')
    def test_cleanup(self, mock_remove, mock_exists):
        """Test cleanup of downloaded files."""
        mock_exists.return_value = True
        
        handler = S3VideoHandler()
        handler.downloaded_files = ["/tmp/file1.mp4", "/tmp/file2.mp4"]
        
        handler.cleanup()
        
        # Verify files were removed
        self.assertEqual(mock_remove.call_count, 2)
        mock_remove.assert_any_call("/tmp/file1.mp4")
        mock_remove.assert_any_call("/tmp/file2.mp4")
        
        # Verify list was cleared
        self.assertEqual(len(handler.downloaded_files), 0)


class TestS3Integration(unittest.TestCase):
    """Integration tests for S3 functionality (requires AWS credentials)."""
    
    def setUp(self):
        """Set up integration tests."""
        # Skip integration tests if AWS credentials are not available
        if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')):
            self.skipTest("AWS credentials not available for integration tests")
    
    def test_real_s3_authentication(self):
        """Test real S3 authentication with environment credentials."""
        client = create_s3_client()
        self.assertIsNotNone(client)
        
        # Test basic S3 operation (list buckets)
        try:
            response = client.list_buckets()
            self.assertIn('Buckets', response)
        except Exception as e:
            self.fail(f"S3 authentication failed: {e}")


def run_tests():
    """Run all S3 video input tests."""
    print("🧪 Running S3 Video Input Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestS3URIValidation,
        TestS3ClientCreation,
        TestS3VideoDownload,
        TestS3VideoHandler,
        TestS3Integration,
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
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
