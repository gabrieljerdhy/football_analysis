#!/usr/bin/env python3
"""
Check what environment variables are being loaded from .env file.
"""

import os
from dotenv import load_dotenv

def check_env_vars():
    """Check the environment variables being loaded."""
    print("🔧 Checking Environment Variables")
    print("=" * 50)
    
    # Load .env file
    load_dotenv()
    
    # Check all DO_SPACES variables
    env_vars = [
        "DO_SPACES_ACCESS_KEY_ID",
        "DO_SPACES_SECRET_ACCESS_KEY", 
        "DO_SPACES_BUCKET",
        "DO_SPACES_REGION",
        "DO_SPACES_ENDPOINT",
        "DO_SPACES_FOLDER_PREFIX"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if var == "DO_SPACES_SECRET_ACCESS_KEY" and value:
            # Mask the secret key for security
            masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
            print(f"{var}: {masked_value}")
        else:
            print(f"{var}: {value}")
    
    print("\n🔍 Checking for any conflicting environment variables...")
    # Check if there are any system environment variables that might override
    for var in env_vars:
        system_value = os.environ.get(var)
        dotenv_value = os.getenv(var)
        if system_value and system_value != dotenv_value:
            print(f"⚠️  System env var {var} differs from .env: {system_value}")

if __name__ == "__main__":
    check_env_vars()
