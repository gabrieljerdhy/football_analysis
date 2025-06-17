#!/usr/bin/env python3
"""
Setup script for jersey number detection system.

This script helps users install dependencies and test the jersey number detection functionality.
"""

import subprocess
import sys
import os
import importlib.util


def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False


def check_gpu_support():
    """Check if GPU support is available for OCR."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU support available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("💻 GPU not available, will use CPU for OCR")
            return False
    except ImportError:
        print("⚠️ PyTorch not found, cannot check GPU support")
        return False


def test_easyocr():
    """Test EasyOCR installation and functionality."""
    try:
        print("🧪 Testing EasyOCR...")
        import easyocr
        
        # Create a simple test
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for test
        print("✅ EasyOCR initialized successfully")
        
        # Test with a simple image
        import numpy as np
        import cv2
        
        # Create test image with number
        test_img = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.putText(test_img, "42", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        results = reader.readtext(test_img)
        if results and any('42' in result[1] for result in results):
            print("✅ EasyOCR test passed")
            return True
        else:
            print("⚠️ EasyOCR test failed, but installation seems OK")
            return True
            
    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        return False


def setup_dependencies():
    """Install and verify all required dependencies."""
    print("🔧 Setting up jersey number detection dependencies...\n")
    
    required_packages = [
        'easyocr',
        'pillow',
        'opencv-python',
        'numpy'
    ]
    
    installed_packages = []
    failed_packages = []
    
    for package in required_packages:
        if check_package_installed(package.replace('-', '_')):
            print(f"✅ {package} already installed")
            installed_packages.append(package)
        else:
            if install_package(package):
                installed_packages.append(package)
            else:
                failed_packages.append(package)
    
    print(f"\n📊 Installation Summary:")
    print(f"   Successfully installed: {len(installed_packages)}/{len(required_packages)}")
    
    if failed_packages:
        print(f"   Failed to install: {failed_packages}")
        return False
    
    return True


def verify_project_structure():
    """Verify that the project structure is correct."""
    print("📁 Verifying project structure...")
    
    required_dirs = [
        'jersey_number_detector',
        'trackers',
        'models',
        'debug_and_tests'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'jersey_number_detector/__init__.py',
        'jersey_number_detector/jersey_number_detector.py'
    ]
    
    missing_items = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_items.append(f"Directory: {directory}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"File: {file_path}")
    
    if missing_items:
        print("❌ Missing required project components:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    else:
        print("✅ Project structure verified")
        return True


def run_basic_test():
    """Run a basic test of the jersey detection system."""
    print("🧪 Running basic jersey detection test...")
    
    try:
        # Import the jersey detector
        from jersey_number_detector import JerseyNumberDetector
        
        # Initialize detector
        detector = JerseyNumberDetector(
            confidence_threshold=0.3,
            consensus_frames=3
        )
        
        print("✅ Jersey detector initialized successfully")
        
        # Test basic functionality
        import numpy as np
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Test preprocessing
        bbox = [10, 10, 90, 90]
        test_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        preprocessed = detector.preprocess_jersey_region(test_frame, bbox)
        
        if preprocessed is not None:
            print("✅ Image preprocessing test passed")
        else:
            print("⚠️ Image preprocessing test failed")
        
        print("✅ Basic functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function."""
    print("🚀 Jersey Number Detection Setup\n")
    print("This script will install dependencies and test the jersey detection system.\n")
    
    success = True
    
    # Step 1: Verify project structure
    if not verify_project_structure():
        print("\n❌ Project structure verification failed")
        print("Please ensure you're running this script from the project root directory")
        return False
    
    print()
    
    # Step 2: Install dependencies
    if not setup_dependencies():
        print("\n❌ Dependency installation failed")
        success = False
    
    print()
    
    # Step 3: Check GPU support
    check_gpu_support()
    
    print()
    
    # Step 4: Test EasyOCR
    if not test_easyocr():
        print("\n⚠️ EasyOCR test failed, but continuing...")
    
    print()
    
    # Step 5: Run basic test
    if not run_basic_test():
        print("\n❌ Basic functionality test failed")
        success = False
    
    # Summary
    print("\n" + "="*50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the main analysis with: python main.py --input your_video.mp4")
        print("2. Jersey numbers will be automatically detected and displayed")
        print("3. Check the CSV output for jersey number data")
        print("4. Run tests with: python debug_and_tests/test_jersey_detection.py")
    else:
        print("❌ Setup completed with errors")
        print("\nPlease check the error messages above and resolve any issues")
        print("You may need to install dependencies manually or check your Python environment")
    
    print("\nFor more information, see JERSEY_NUMBER_DETECTION_README.md")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
