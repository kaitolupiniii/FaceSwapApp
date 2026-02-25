"""
Virtual camera output module for Windows
"""
import cv2
import numpy as np
import pyvirtualcam
import threading
import queue
from utils import logger
import config

class VirtualCamera:
    def __init__(self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS):
        """Initialize virtual camera"""
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera = None
        self.output_thread = None
        
        logger.info(f"VirtualCamera initialized with {width}x{height} @ {fps}fps")
    
    def start(self):
        """Start virtual camera output"""
        try:
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                backend='obs'  # Use OBS virtual camera
            )
            self.running = True
            self.output_thread = threading.Thread(target=self._output_loop)
            self.output_thread.daemon = True
            self.output_thread.start()
            logger.info("Virtual camera started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start virtual camera: {e}")
            return False
    
    def _output_loop(self):
        """Main output loop"""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                # Ensure frame is correct size
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert BGR to RGB (pyvirtualcam expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Send to virtual camera
                self.camera.send(frame_rgb)
                
                # Wait for next frame
                self.camera.sleep_until_next_frame()
                
            except queue.Empty:
                # Send black frame if no input
                black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.camera.send(black_frame)
                
            except Exception as e:
                logger.error(f"Error in output loop: {e}")
    
    def send_frame(self, frame):
        """Send frame to virtual camera"""
        if self.running:
            try:
                # Non-blocking put
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip frame if queue is full
    
    def stop(self):
        """Stop virtual camera"""
        self.running = False
        if self.output_thread:
            self.output_thread.join(timeout=2.0)
        if self.camera:
            self.camera.close()
        logger.info("Virtual camera stopped")