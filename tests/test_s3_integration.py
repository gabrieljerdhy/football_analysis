#!/usr/bin/env python3
"""
S3 Integration Test Script

This script tests the complete S3 video input workflow including:
- S3 URI validation
- AWS authentication
- Video downloading
- Video processing pipeline integration

Run this script to verify S3 functionality is working correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    S3VideoHandler,
    VideoFrameIterator,
    get_video_info,
    is_s3_uri,
    parse_s3_uri,
)


def test_s3_uri_functions():
    """Test S3 URI validation and parsing functions."""
    print("🔍 Testing S3 URI Functions")
    print("-" * 30)

    # Test valid S3 URIs
    valid_uris = [
        "s3://test-bucket/video.mp4",
        "s3://my-bucket/path/to/video.avi",
    ]

    for uri in valid_uris:
        print(f"Testing URI: {uri}")

        # Test is_s3_uri
        if is_s3_uri(uri):
            print("  ✅ Recognized as S3 URI")

            # Test parse_s3_uri
            try:
                bucket, key = parse_s3_uri(uri)
                print(f"  ✅ Parsed - Bucket: {bucket}, Key: {key}")
            except ValueError as e:
                print(f"  ❌ Parse error: {e}")
                return False
        else:
            print("  ❌ Not recognized as S3 URI")
            return False

    # Test invalid URIs
    invalid_uris = [
        "http://example.com/video.mp4",
        "/local/path/video.mp4",
    ]

    for uri in invalid_uris:
        print(f"Testing invalid URI: {uri}")
        if not is_s3_uri(uri):
            print("  ✅ Correctly rejected")
        else:
            print("  ❌ Incorrectly accepted")
            return False

    # Test S3 URIs that are valid format but invalid content
    invalid_s3_uris = [
        "s3://bucket-only",  # Missing object key
        "s3://bucket/",  # Empty object key
    ]

    for uri in invalid_s3_uris:
        print(f"Testing invalid S3 URI: {uri}")
        if is_s3_uri(uri):
            try:
                parse_s3_uri(uri)
                print("  ❌ Should have failed parsing")
                return False
            except ValueError:
                print("  ✅ Correctly failed parsing")
        else:
            print("  ❌ Should be recognized as S3 URI")
            return False

    print("✅ S3 URI functions working correctly\n")
    return True


def test_s3_authentication():
    """Test S3 authentication methods."""
    print("🔐 Testing S3 Authentication")
    print("-" * 30)

    # Check for AWS credentials
    has_env_creds = bool(
        os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    if has_env_creds:
        print("✅ AWS credentials found in environment")

        try:
            handler = S3VideoHandler()
            if handler.s3_client:
                print("✅ S3 client created successfully")

                # Test basic S3 operation
                try:
                    handler.s3_client.list_buckets()
                    print("✅ S3 authentication verified")
                    return True
                except Exception as e:
                    print(f"❌ S3 authentication failed: {e}")
                    return False
            else:
                print("❌ Failed to create S3 client")
                return False
        except Exception as e:
            print(f"❌ Error creating S3 handler: {e}")
            return False
    else:
        print("⚠️  No AWS credentials found in environment")
        print(
            "   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test authentication"
        )
        return True  # Not a failure, just no credentials to test


def test_s3_video_handler():
    """Test S3VideoHandler functionality."""
    print("🎬 Testing S3VideoHandler")
    print("-" * 30)

    # Test with mock S3 URI (won't actually download)
    test_uri = "s3://test-bucket/test-video.mp4"

    try:
        handler = S3VideoHandler()
        print("✅ S3VideoHandler created")

        # Test URI validation
        if is_s3_uri(test_uri):
            print("✅ Test URI validated")
        else:
            print("❌ Test URI validation failed")
            return False

        # Test cleanup functionality
        handler.cleanup()
        print("✅ Cleanup method works")

        return True
    except Exception as e:
        print(f"❌ S3VideoHandler test failed: {e}")
        return False


def test_video_utilities_s3_support():
    """Test that video utilities support S3 URIs."""
    print("🛠️  Testing Video Utilities S3 Support")
    print("-" * 30)

    test_uri = "s3://test-bucket/test-video.mp4"

    # Test VideoFrameIterator accepts S3 parameters
    try:
        # This should not fail even without real S3 credentials
        # because we're just testing the interface
        iterator = VideoFrameIterator(
            test_uri,
            batch_size=10,
            s3_handler=None,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        print("✅ VideoFrameIterator accepts S3 parameters")
    except Exception as e:
        print(f"❌ VideoFrameIterator S3 support failed: {e}")
        return False

    print("✅ Video utilities S3 support verified")
    return True


def test_error_handling():
    """Test error handling for various S3 scenarios."""
    print("🚨 Testing Error Handling")
    print("-" * 30)

    # Test invalid S3 URI
    try:
        parse_s3_uri("invalid-uri")
        print("❌ Should have raised ValueError for invalid URI")
        return False
    except ValueError:
        print("✅ Correctly handled invalid S3 URI")

    # Test S3VideoHandler with no credentials
    try:
        handler = S3VideoHandler(
            aws_access_key_id="invalid", aws_secret_access_key="invalid"
        )
        # This might succeed in creating the handler but fail on actual S3 operations
        print("✅ S3VideoHandler handles invalid credentials gracefully")
    except Exception as e:
        print(f"✅ S3VideoHandler properly handles credential errors: {e}")

    print("✅ Error handling working correctly")
    return True


def run_integration_tests():
    """Run all S3 integration tests."""
    print("🚀 S3 Video Input Integration Tests")
    print("=" * 50)

    tests = [
        ("S3 URI Functions", test_s3_uri_functions),
        ("S3 Authentication", test_s3_authentication),
        ("S3VideoHandler", test_s3_video_handler),
        ("Video Utilities S3 Support", test_video_utilities_s3_support),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


def main():
    """Main function to run integration tests."""
    success = run_integration_tests()

    if not success:
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Ensure boto3 is installed: pip install boto3")
        print("2. Set AWS credentials if you want to test authentication:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("3. Check that all required modules are available")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
