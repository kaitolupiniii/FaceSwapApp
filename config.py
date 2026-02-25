"""
Configuration file for Face Swap Application
"""
import os

# Application settings
APP_NAME = "Real-Time Face Swap"
VERSION = "1.0.0"

# Camera settings
CAMERA_ID = 0  # Default camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Face detection settings
DETECTION_CONFIDENCE = 0.5
MAX_FACES = 2
USE_GPU = True

# Face swap settings
BLEND_ALPHA = 0.7  # Blending factor
SMOOTHING_FACTOR = 0.8  # Temporal smoothing
COLOR_CORRECTION = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, "source_faces")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for dir_path in [MODELS_DIR, SOURCE_IMAGES_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)