"""
Face detection and landmark extraction module
"""
import cv2
import mediapipe as mp
import numpy as np
from utils import logger
import config

class FaceDetector:
    def __init__(self):
        """Initialize face detector with MediaPipe"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for long-range
            min_detection_confidence=config.DETECTION_CONFIDENCE
        )
        
        # Initialize face mesh for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_FACES,
            refine_landmarks=True,  # Get more accurate landmarks
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        
        # Face tracking for temporal smoothing
        self.prev_landmarks = None
        self.smoothing_factor = config.SMOOTHING_FACTOR
        
        logger.info("FaceDetector initialized successfully")
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        Returns: list of face bounding boxes
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure box is within frame
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'score': detection.score[0]
                })
        
        return faces
    
    def get_face_landmarks(self, frame, bbox=None):
        """
        Extract facial landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = frame.shape[:2]
        
        # Get landmarks for first face
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        # Apply temporal smoothing
        if self.prev_landmarks is not None:
            smoothed = []
            for curr, prev in zip(landmarks, self.prev_landmarks):
                x = int(self.smoothing_factor * prev[0] + (1 - self.smoothing_factor) * curr[0])
                y = int(self.smoothing_factor * prev[1] + (1 - self.smoothing_factor) * curr[1])
                smoothed.append((x, y))
            self.prev_landmarks = smoothed
            return np.array(smoothed)
        else:
            self.prev_landmarks = landmarks
            return np.array(landmarks)
    
    def get_face_orientation(self, landmarks):
        """
        Estimate face orientation (yaw, pitch, roll)
        """
        # Get key points
        left_eye = np.mean([landmarks[33], landmarks[133]], axis=0)
        right_eye = np.mean([landmarks[362], landmarks[263]], axis=0)
        nose = landmarks[1]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Calculate angles
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Simplified orientation estimation
        yaw = np.arctan2(nose[0] - eye_center[0], nose[1] - eye_center[1])
        pitch = np.arctan2(nose[1] - eye_center[1], nose[2] if len(nose) > 2 else 1)
        
        return {'yaw': yaw, 'pitch': pitch}
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw landmarks on frame (for debugging)
        """
        for point in landmarks:
            cv2.circle(frame, tuple(point.astype(int)), 1, (0, 255, 0), -1)
        return frame