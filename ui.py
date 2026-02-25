"""
User interface module using PyQt5
"""
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QSlider, QCheckBox, QGroupBox, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from utils import logger
import config

class VideoThread(QThread):
    """Thread for video capture"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.camera = cv2.VideoCapture(config.CAMERA_ID)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.FPS)
        
        # Processing flag
        self.process_frame = True
        
    def run(self):
        """Main loop"""
        while self.running:
            ret, frame = self.camera.read()
            if ret and self.process_frame:
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.change_pixmap_signal.emit(frame_rgb)
        
        self.camera.release()
    
    def stop(self):
        """Stop thread"""
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        from face_detector import FaceDetector
        from face_swapper import FaceSwapper
        from virtual_camera import VirtualCamera
        
        self.face_detector = FaceDetector()
        self.face_swapper = FaceSwapper()
        self.virtual_camera = VirtualCamera()
        
        self.source_face_loaded = False
        self.face_swap_enabled = False
        
        # Setup UI
        self._setup_ui()
        
        # Setup video thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()
        
        # Setup timer for processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(33)  # ~30 FPS
        
        # Current frame
        self.current_frame = None
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Setup user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black")
        left_layout.addWidget(self.video_label)
        
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMaximumWidth(300)
        
        # Source face group
        source_group = QGroupBox("Source Face")
        source_layout = QVBoxLayout()
        
        self.load_source_btn = QPushButton("Load Source Face")
        self.load_source_btn.clicked.connect(self.load_source_face)
        source_layout.addWidget(self.load_source_btn)
        
        self.source_status = QLabel("No source face loaded")
        source_layout.addWidget(self.source_status)
        
        source_group.setLayout(source_layout)
        right_layout.addWidget(source_group)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        self.enable_swap_check = QCheckBox("Enable Face Swap")
        self.enable_swap_check.stateChanged.connect(self.toggle_face_swap)
        controls_layout.addWidget(self.enable_swap_check)
        
        # Blend slider
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("Blend Alpha:"))
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(int(config.BLEND_ALPHA * 100))
        self.blend_slider.valueChanged.connect(self.update_blend)
        blend_layout.addWidget(self.blend_slider)
        controls_layout.addLayout(blend_layout)
        
        # Smoothing slider
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothing:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(int(config.SMOOTHING_FACTOR * 100))
        self.smooth_slider.valueChanged.connect(self.update_smoothing)
        smooth_layout.addWidget(self.smooth_slider)
        controls_layout.addLayout(smooth_layout)
        
        # Color correction checkbox
        self.color_correct_check = QCheckBox("Color Correction")
        self.color_correct_check.setChecked(config.COLOR_CORRECTION)
        self.color_correct_check.stateChanged.connect(self.toggle_color_correction)
        controls_layout.addWidget(self.color_correct_check)
        
        controls_group.setLayout(controls_layout)
        right_layout.addWidget(controls_group)
        
        # Camera group
        camera_group = QGroupBox("Virtual Camera")
        camera_layout = QVBoxLayout()
        
        self.start_camera_btn = QPushButton("Start Virtual Camera")
        self.start_camera_btn.clicked.connect(self.toggle_virtual_camera)
        camera_layout.addWidget(self.start_camera_btn)
        
        self.camera_status = QLabel("Virtual camera: Stopped")
        camera_layout.addWidget(self.camera_status)
        
        camera_group.setLayout(camera_layout)
        right_layout.addWidget(camera_group)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0")
        right_layout.addWidget(self.fps_label)
        
        # Add stretch to push everything up
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel, 1)
    
    def update_image(self, frame):
        """Update video display"""
        self.current_frame = frame
    
    def process_frame(self):
        """Process current frame"""
        if self.current_frame is None:
            return
        
        import time
        start_time = time.time()
        
        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        
        if self.face_swap_enabled and self.source_face_loaded:
            # Detect faces
            faces = self.face_detector.detect_faces(frame_bgr)
            
            if faces:
                # Get landmarks for first face
                landmarks = self.face_detector.get_face_landmarks(frame_bgr)
                
                if landmarks is not None:
                    # Perform face swap
                    frame_bgr = self.face_swapper.swap_face(frame_bgr, landmarks)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time) if start_time else 0
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Send to virtual camera if running
        if hasattr(self, 'virtual_camera') and self.virtual_camera.running:
            self.virtual_camera.send_frame(frame_bgr)
        
        # Convert back to RGB for display
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Display frame
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
    
    def load_source_face(self):
        """Load source face image"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Source Face", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            success = self.face_swapper.load_source_face(file_name)
            if success:
                self.source_face_loaded = True
                self.source_status.setText(f"Loaded: {file_name.split('/')[-1]}")
                self.source_status.setStyleSheet("color: green")
            else:
                self.source_status.setText("Failed to load image")
                self.source_status.setStyleSheet("color: red")
    
    def toggle_face_swap(self, state):
        """Toggle face swap processing"""
        self.face_swap_enabled = (state == Qt.Checked)
        self.video_thread.process_frame = self.face_swap_enabled
    
    def update_blend(self, value):
        """Update blend alpha value"""
        config.BLEND_ALPHA = value / 100.0
    
    def update_smoothing(self, value):
        """Update smoothing factor"""
        config.SMOOTHING_FACTOR = value / 100.0
        if hasattr(self, 'face_detector'):
            self.face_detector.smoothing_factor = config.SMOOTHING_FACTOR
    
    def toggle_color_correction(self, state):
        """Toggle color correction"""
        config.COLOR_CORRECTION = (state == Qt.Checked)
    
    def toggle_virtual_camera(self):
        """Start/stop virtual camera"""
        if not hasattr(self, 'virtual_camera') or not self.virtual_camera.running:
            if self.virtual_camera.start():
                self.start_camera_btn.setText("Stop Virtual Camera")
                self.camera_status.setText("Virtual camera: Running")
                self.camera_status.setStyleSheet("color: green")
        else:
            self.virtual_camera.stop()
            self.start_camera_btn.setText("Start Virtual Camera")
            self.camera_status.setText("Virtual camera: Stopped")
            self.camera_status.setStyleSheet("color: red")
    
    def closeEvent(self, event):
        """Handle window close"""
        self.video_thread.stop()
        if hasattr(self, 'virtual_camera') and self.virtual_camera.running:
            self.virtual_camera.stop()
        event.accept()