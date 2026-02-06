import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional

class ImagePreprocessor:
    """
    Layer 1: Image Preprocessing
    Clean the scan before OCR with denoising, deskew, adaptive thresholding, etc.
    """
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {ext}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce image noise"""
        try:
            # Use Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
        except Exception as e:
            print(f"Error in denoising: {e}")
            return image
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            # Use median angle for deskewing
            median_angle = np.median(angles)
            
            # Rotate image to correct skew
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            deskewed = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return deskewed
        except Exception as e:
            print(f"Error in deskewing: {e}")
            return image
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            # For better results with varying lighting conditions
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to BGR for consistency
            adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            return adaptive_bgr
        except Exception as e:
            print(f"Error in adaptive thresholding: {e}")
            return image
    
    def normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        """Normalize contrast and brightness"""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to the Y channel
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            
            # Convert back to BGR
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return normalized
        except Exception as e:
            print(f"Error in contrast normalization: {e}")
            return image
    
    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove unnecessary borders from the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find the largest contour (should be the main content)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            return cropped
        except Exception as e:
            print(f"Error in border removal: {e}")
            return image
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary format"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to BGR for consistency
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return binary_bgr
        except Exception as e:
            print(f"Error in binarization: {e}")
            return image
    
    def preprocess(self, image_path: str, output_path: str = None) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline
        Returns processed image and processing metadata
        """
        try:
            # Load image
            image = self.load_image(image_path)
            if image is None:
                return None, {"error": "Failed to load image"}
            
            original_shape = image.shape
            
            processed = image.copy()
            
            processed = self.denoise(processed)
            
            processed = self.deskew(processed)
            
            processed = self.normalize_contrast(processed)
            
            processed = self.remove_borders(processed)
            
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            binary = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
       
            if output_path:
                cv2.imwrite(output_path, binary)
            
            metadata = {
                "original_shape": original_shape,
                "processed_shape": binary.shape,
                "steps_applied": ["denoise", "deskew", "normalize_contrast", "remove_borders", "adaptive_threshold"],
                "status": "success"
            }
            
            return binary, metadata
            
        except Exception as e:
            error_msg = f"Error in preprocessing pipeline: {e}"
            print(error_msg)
            return None, {"error": error_msg}
    
    def preprocess_batch(self, image_paths: list, output_dir: str) -> list:
        """Process multiple images"""
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            output_path = os.path.join(output_dir, f"processed_{i+1}.png")
            processed_image, metadata = self.preprocess(image_path, output_path)
            results.append({
                "input_path": image_path,
                "output_path": output_path,
                "metadata": metadata
            })
        
        return results
