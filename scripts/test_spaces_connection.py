#!/usr/bin/env python3
"""
Test script to debug DigitalOcean Spaces connection issues.
"""

import os
from utils.storage_utils import create_uploader_from_env

def test_connection():
    """Test the DigitalOcean Spaces connection with detailed debugging."""
    print("🔧 Testing DigitalOcean Spaces Connection")
    print("=" * 50)
    
    # Create uploader from .env
    uploader = create_uploader_from_env()
    
    if not uploader:
        print("❌ Failed to create uploader")
        return False
    
    print(f"✅ Uploader created successfully")
    print(f"📍 Region: {uploader.region}")
    print(f"🔗 Endpoint: {uploader.endpoint_url}")
    print(f"🪣 Default bucket: {uploader.bucket_name}")
    
    # Test connection to the bucket
    bucket_name = os.getenv("DO_SPACES_BUCKET")
    if bucket_name:
        print(f"\n🧪 Testing connection to bucket: {bucket_name}")
        success = uploader.test_connection(bucket_name)
        
        if success:
            print("✅ Connection test successful!")
            
            # Try to list objects in the bucket (optional)
            try:
                response = uploader.client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                print(f"✅ Bucket access confirmed - found {response.get('KeyCount', 0)} objects")
            except Exception as e:
                print(f"⚠️  Bucket accessible but couldn't list objects: {e}")
                
        else:
            print("❌ Connection test failed")
            
            # Try to list all buckets to see what's available
            try:
                print("\n🔍 Trying to list available buckets...")
                response = uploader.client.list_buckets()
                buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
                if buckets:
                    print(f"📋 Available buckets: {', '.join(buckets)}")
                else:
                    print("📋 No buckets found")
            except Exception as e:
                print(f"❌ Couldn't list buckets: {e}")
                
        return success
    else:
        print("❌ No bucket name found in environment")
        return False

if __name__ == "__main__":
    test_connection()
