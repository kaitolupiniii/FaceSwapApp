"""
Setup script for Face Swap Application
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

def check_python_version():
    """Check Python version"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]}+ required")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def download_shape_predictor():
    """Download dlib shape predictor"""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    predictor_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
    
    if not os.path.exists(predictor_path):
        print("Downloading face landmark predictor...")
        bz2_path = predictor_path + ".bz2"
        
        # Download
        urllib.request.urlretrieve(url, bz2_path)
        
        # Extract
        import bz2
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(predictor_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Clean up
        os.remove(bz2_path)
        print("Download complete")
    else:
        print("Face landmark predictor already exists")
    
    return True

def setup_virtual_camera():
    """Setup virtual camera for Windows"""
    print("\nVirtual Camera Setup:")
    print("Option 1: Install OBS Studio (Recommended)")
    print("   Download from: https://obsproject.com/download")
    print("\nOption 2: Use pyvirtualcam with OBS Virtual Camera plugin")
    print("   Install OBS first, then the plugin will be available")
    print("\nAfter installing OBS:")
    print("1. Open OBS")
    print("2. Go to Tools -> VirtualCam")
    print("3. Click Start")
    print("4. The virtual camera will be available in your meeting apps")
    
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("Face Swap Application Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        return
    
    # Install requirements
    if not install_requirements():
        input("\nPress Enter to exit...")
        return
    
    # Download models
    if not download_shape_predictor():
        input("\nPress Enter to exit...")
        return
    
    # Setup virtual camera
    setup_virtual_camera()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nTo run the application:")
    print("  python main.py")
    print("\nMake sure to:")
    print("1. Load a source face image")
    print("2. Enable face swap")
    print("3. Start virtual camera")
    print("4. Select virtual camera in your meeting app")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()