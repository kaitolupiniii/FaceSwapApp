"""
Face swapping module
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay
from utils import (
    get_triangles, morph_triangle, create_face_mask,
    apply_color_transfer, logger
)
import config

class FaceSwapper:
    def __init__(self, source_image_path=None):
        """Initialize face swapper with source face"""
        self.source_face = None
        self.source_landmarks = None
        self.source_triangles = None
        self.target_face_history = []
        self.history_length = 5
        
        if source_image_path:
            self.load_source_face(source_image_path)
        
        logger.info("FaceSwapper initialized")
    
    def load_source_face(self, image_path):
        """
        Load and process source face image
        """
        try:
            # Load image
            self.source_face = cv2.imread(image_path)
            if self.source_face is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect landmarks (assuming we have face detector)
            # This would need integration with FaceDetector
            # For now, we'll use a placeholder
            self.source_landmarks = self._get_placeholder_landmarks()
            
            # Calculate Delaunay triangles
            h, w = self.source_face.shape[:2]
            rect = (0, 0, w, h)
            self.source_triangles = get_triangles(rect, self.source_landmarks)
            
            logger.info(f"Source face loaded from {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading source face: {e}")
            return False
    
    def _get_placeholder_landmarks(self):
        """
        Generate placeholder landmarks (for testing)
        In production, use actual face detection
        """
        if self.source_face is None:
            return None
        
        h, w = self.source_face.shape[:2]
        
        # Generate key facial points (simplified)
        landmarks = []
        
        # Jaw line (17 points)
        for i in range(17):
            x = int(w * (0.2 + 0.6 * i / 16))
            y = int(h * (0.7 + 0.1 * np.sin(i * np.pi / 8)))
            landmarks.append([x, y])
        
        # Left eyebrow (5 points)
        for i in range(5):
            x = int(w * (0.3 + 0.1 * i))
            y = int(h * 0.3)
            landmarks.append([x, y])
        
        # Right eyebrow (5 points)
        for i in range(5):
            x = int(w * (0.6 + 0.1 * i))
            y = int(h * 0.3)
            landmarks.append([x, y])
        
        # Nose (9 points)
        for i in range(9):
            x = int(w * (0.5 + 0.02 * (i - 4)))
            y = int(h * (0.4 + 0.04 * i))
            landmarks.append([x, y])
        
        # Left eye (6 points)
        left_eye_center = (int(w * 0.35), int(h * 0.35))
        for angle in np.linspace(0, 2*np.pi, 6):
            x = int(left_eye_center[0] + 10 * np.cos(angle))
            y = int(left_eye_center[1] + 10 * np.sin(angle))
            landmarks.append([x, y])
        
        # Right eye (6 points)
        right_eye_center = (int(w * 0.65), int(h * 0.35))
        for angle in np.linspace(0, 2*np.pi, 6):
            x = int(right_eye_center[0] + 10 * np.cos(angle))
            y = int(right_eye_center[1] + 10 * np.sin(angle))
            landmarks.append([x, y])
        
        # Lips (20 points)
        lip_center = (int(w * 0.5), int(h * 0.6))
        for angle in np.linspace(0, 2*np.pi, 20):
            x = int(lip_center[0] + 20 * np.cos(angle))
            y = int(lip_center[1] + 10 * np.sin(angle))
            landmarks.append([x, y])
        
        return np.array(landmarks, dtype=np.int32)
    
    def swap_face(self, target_frame, target_landmarks):
        """
        Swap source face onto target frame
        """
        if self.source_face is None or self.source_landmarks is None:
            return target_frame
        
        try:
            # Create copy of target frame
            result = target_frame.copy()
            
            # Get face mask
            face_mask = create_face_mask(target_landmarks, target_frame.shape)
            
            # Warp source face to match target landmarks
            warped_face = self._warp_source_to_target(target_landmarks)
            
            if warped_face is None:
                return target_frame
            
            # Color correction
            if config.COLOR_CORRECTION:
                warped_face = apply_color_transfer(
                    self.source_face, 
                    warped_face
                )
            
            # Blend faces
            mask_3channel = cv2.merge([face_mask, face_mask, face_mask])
            
            # Apply seamless cloning for better blending
            try:
                # Find face center
                center = np.mean(target_landmarks, axis=0).astype(int)
                
                # Apply seamless cloning
                result = cv2.seamlessClone(
                    warped_face, target_frame, 
                    (mask_3channel * 255).astype(np.uint8),
                    tuple(center),
                    cv2.NORMAL_CLONE
                )
            except:
                # Fallback to alpha blending
                result = (1 - mask_3channel * config.BLEND_ALPHA) * target_frame + \
                        mask_3channel * config.BLEND_ALPHA * warped_face
                result = result.astype(np.uint8)
            
            # Update history for temporal consistency
            self._update_history(result, target_landmarks)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in face swap: {e}")
            return target_frame
    
    def _warp_source_to_target(self, target_landmarks):
        """
        Warp source face to match target landmarks using Delaunay triangulation
        """
        h, w = self.source_face.shape[:2]
        
        # Create output image
        warped = np.zeros_like(self.source_face)
        
        # Get Delaunay triangles
        rect = (0, 0, w, h)
        triangles = get_triangles(rect, self.source_landmarks)
        
        # Warp each triangle
        for t in triangles:
            # Get triangle vertices
            t_src = []
            t_tgt = []
            
            for point in t:
                # Find closest landmarks
                src_idx = self._find_closest_landmark(point, self.source_landmarks)
                tgt_idx = self._find_closest_landmark(point, target_landmarks)
                
                if src_idx is not None and tgt_idx is not None:
                    t_src.append(self.source_landmarks[src_idx])
                    t_tgt.append(target_landmarks[tgt_idx])
            
            if len(t_src) == 3 and len(t_tgt) == 3:
                # Morph triangle
                morph_triangle(
                    self.source_face, 
                    np.zeros_like(self.source_face),
                    warped,
                    t_src,
                    t_tgt,
                    t_src,  # Use source points for warping
                    1.0  # Full warp to target
                )
        
        # Crop to face region
        face_rect = cv2.boundingRect(target_landmarks)
        x, y, w, h = face_rect
        warped = warped[y:y+h, x:x+w]
        
        return warped
    
    def _find_closest_landmark(self, point, landmarks, threshold=50):
        """
        Find closest landmark to point
        """
        distances = [np.linalg.norm(np.array(point) - np.array(landmark)) 
                    for landmark in landmarks]
        min_dist = min(distances)
        
        if min_dist < threshold:
            return np.argmin(distances)
        return None
    
    def _update_history(self, frame, landmarks):
        """
        Update face history for temporal smoothing
        """
        self.target_face_history.append({
            'frame': frame.copy(),
            'landmarks': landmarks.copy()
        })
        
        if len(self.target_face_history) > self.history_length:
            self.target_face_history.pop(0)
    
    def get_temporal_smooth(self, current_frame, current_landmarks):
        """
        Apply temporal smoothing using history
        """
        if len(self.target_face_history) < 2:
            return current_frame
        
        # Average with previous frames
        avg_frame = current_frame.astype(np.float32)
        weight_sum = 1.0
        
        for i, hist in enumerate(reversed(self.target_face_history[:-1])):
            weight = 0.5 ** (i + 1)  # Exponential decay
            avg_frame += hist['frame'].astype(np.float32) * weight
            weight_sum += weight
        
        avg_frame /= weight_sum
        
        return avg_frame.astype(np.uint8)