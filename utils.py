"""
Utility functions for face swap application
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay
import logging
from datetime import datetime
import os
from config import LOG_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'face_swap_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Apply affine transform to triangle
    """
    # Get affine transformation matrix
    warp_mat = cv2.getAffineTransform(
        np.float32(src_tri), 
        np.float32(dst_tri)
    )
    
    # Apply transformation
    dst = cv2.warpAffine(
        src, warp_mat, (size[0], size[1]),
        None, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return dst

def morph_triangle(img1, img2, img_warped, t1, t2, t, alpha):
    """
    Morph between two triangles
    """
    # Get bounding rectangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of respective rectangles
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply affine transform to source and target images
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    # Alpha blend
    img_rect = (1 - alpha) * warp_img1 + alpha * warp_img2

    # Copy triangle region to output image
    img_warped[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
        img_warped[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask

def get_triangles(rect, points):
    """
    Calculate Delaunay triangles for face mesh
    """
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points
    for p in points:
        subdiv.insert(p)
    
    # Get triangles
    triangle_list = subdiv.getTriangleList()
    
    # Convert to point triplets
    triangles = []
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        triangles.append([pt1, pt2, pt3])
    
    return triangles

def constrain_point(point, bounds):
    """
    Constrain point within image bounds
    """
    x = max(0, min(point[0], bounds[1]))
    y = max(0, min(point[1], bounds[0]))
    return (int(x), int(y))

def resize_image(image, width=None, height=None):
    """
    Resize image maintaining aspect ratio
    """
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def apply_color_transfer(source, target):
    """
    Transfer color from source to target image
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    
    # Calculate statistics
    for i in range(3):
        # Source statistics
        src_mean = np.mean(source_lab[:, :, i])
        src_std = np.std(source_lab[:, :, i])
        
        # Target statistics
        tgt_mean = np.mean(target_lab[:, :, i])
        tgt_std = np.std(target_lab[:, :, i])
        
        # Transfer
        target_lab[:, :, i] = ((target_lab[:, :, i] - tgt_mean) * 
                               (src_std / tgt_std)) + src_mean
    
    # Clip and convert back
    target_lab = np.clip(target_lab, 0, 255)
    result = cv2.cvtColor(target_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    return result

def create_face_mask(landmarks, shape):
    """
    Create a face mask from landmarks
    """
    mask = np.zeros(shape[:2], dtype=np.float32)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, 1.0)
    
    # Feather the edges
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    return mask

def calculate_fps(prev_time):
    """
    Calculate FPS
    """
    import time
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    return fps, current_time