import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import os

class RegionSegmentation:
    def __init__(self, model_path: str = None):
        self.region_types = [
            "investor_section",
            "transaction_section", 
            "personal_details",
            "financial_details",
            "signature",
            "header",
            "table",
            "paragraph"
        ]
        
        # Always try to load YOLO
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ Loaded custom YOLO model from: {model_path}")
            else:
                # Use detection model instead of segmentation
                self.model = YOLO('yolov8n.pt')  # Detection model
                print("✅ Using YOLOv8n detection model (segmentation model not available)")
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            self.model = None
    
    def segment_document(self, image: np.ndarray) -> Dict:
        """
        Improved segmentation with better fallback methods
        """
        try:
            # Method 1: Try YOLO detection first
            if self.model is not None:
                try:
                    # Run detection
                    results = self.model(image)
                    
                    regions = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                
                                # Skip if confidence is too low
                                if conf < 0.3:
                                    continue
                                
                                region_type = self.region_types[cls] if cls < len(self.region_types) else "unknown"
                                
                                regions.append({
                                    "bounding_box": {
                                        "x": int(x1),
                                        "y": int(y1),
                                        "width": int(x2 - x1),
                                        "height": int(y2 - y1)
                                    },
                                    "region_type": region_type,
                                    "confidence": float(conf)
                                })
                    
                    if regions:
                        print(f"✅ YOLO detection found {len(regions)} regions")
                        return {
                            "regions": regions,
                            "detection_method": "yolo_detection",
                            "status": "success"
                        }
                except Exception as e:
                    print(f"YOLO detection failed: {e}")
            
            # Method 2: Text-based segmentation (most reliable for documents)
            return self._text_based_segmentation(image)
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return self._full_document_fallback(image)
    def _text_based_segmentation(self, image: np.ndarray) -> Dict:
        """Segment document based on text density"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Morphological operations to connect text areas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            img_h, img_w = image.shape[:2]
            min_area = img_h * img_w * 0.005  # 0.5% of image area
            max_area = img_h * img_w * 0.8    # 80% of image area
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter very small regions
                    if w < 30 or h < 30:
                        continue
                    
                    # Extract region image to check if it contains text
                    region_img = gray[y:y+h, x:x+w]
                    
                    # Check if region contains significant text
                    if self._contains_text(region_img):
                        # Classify region type
                        region_type = self._classify_region_by_shape(x, y, w, h, image.shape)
                        
                        regions.append({
                            "bounding_box": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h
                            },
                            "region_type": region_type,
                            "confidence": 0.7
                        })
            
            # Merge overlapping regions
            regions = self._merge_overlapping_regions(regions)
            
            if regions:
                print(f"✅ Text-based segmentation found {len(regions)} regions")
                return {
                    "regions": regions,
                    "detection_method": "text_based",
                    "status": "success"
                }
            else:
                return self._full_document_fallback(image)
                
        except Exception as e:
            print(f"Text-based segmentation error: {e}")
            return self._full_document_fallback(image)
    
    def _contains_text(self, region_img: np.ndarray) -> bool:
        """Check if a region contains significant text"""
        try:
            if region_img.size == 0:
                return False
            
            # Calculate text density
            edges = cv2.Canny(region_img, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate horizontal projection
            horizontal_proj = np.sum(region_img < 128, axis=1)
            var_horiz = np.var(horizontal_proj)
            
            # Heuristic: regions with edges and variation likely contain text
            return edge_density > 0.05 and var_horiz > 10
            
        except Exception:
            return False
    
    def _merge_overlapping_regions(self, regions: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping regions"""
        if not regions:
            return []
        
        merged_regions = []
        
        for region in regions:
            merged = False
            
            for i, merged_region in enumerate(merged_regions):
                # Calculate overlap
                x1 = max(region["bounding_box"]["x"], merged_region["bounding_box"]["x"])
                y1 = max(region["bounding_box"]["y"], merged_region["bounding_box"]["y"])
                x2 = min(region["bounding_box"]["x"] + region["bounding_box"]["width"], 
                        merged_region["bounding_box"]["x"] + merged_region["bounding_box"]["width"])
                y2 = min(region["bounding_box"]["y"] + region["bounding_box"]["height"], 
                        merged_region["bounding_box"]["y"] + merged_region["bounding_box"]["height"])
                
                if x2 > x1 and y2 > y1:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    region_area = region["bounding_box"]["width"] * region["bounding_box"]["height"]
                    
                    if overlap_area / region_area > overlap_threshold:
                        # Merge regions
                        merged_regions[i]["bounding_box"]["x"] = min(region["bounding_box"]["x"], 
                                                                    merged_region["bounding_box"]["x"])
                        merged_regions[i]["bounding_box"]["y"] = min(region["bounding_box"]["y"], 
                                                                    merged_region["bounding_box"]["y"])
                        merged_regions[i]["bounding_box"]["width"] = max(
                            region["bounding_box"]["x"] + region["bounding_box"]["width"],
                            merged_region["bounding_box"]["x"] + merged_region["bounding_box"]["width"]
                        ) - merged_regions[i]["bounding_box"]["x"]
                        merged_regions[i]["bounding_box"]["height"] = max(
                            region["bounding_box"]["y"] + region["bounding_box"]["height"],
                            merged_region["bounding_box"]["y"] + merged_region["bounding_box"]["height"]
                        ) - merged_regions[i]["bounding_box"]["y"]
                        
                        # Use higher confidence
                        merged_regions[i]["confidence"] = max(region["confidence"], 
                                                            merged_region["confidence"])
                        merged = True
                        break
            
            if not merged:
                merged_regions.append(region)
        
        return merged_regions
    
    def _classify_region_by_shape(self, x: int, y: int, w: int, h: int, image_shape: tuple) -> str:
        """Classify region based on geometric properties"""
        img_h, img_w = image_shape[:2]
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Position in document
        is_top = y < img_h * 0.3
        is_bottom = y > img_h * 0.7
        is_left = x < img_w * 0.3
        
        # Classification logic
        if h < img_h * 0.1 and w > img_w * 0.5 and is_top:
            return "header"
        elif aspect_ratio > 3 and w > img_w * 0.6:
            return "table"
        elif is_bottom and h < img_h * 0.15:
            return "signature"
        elif w < img_w * 0.4 and h > img_h * 0.2:
            return "personal_details"
        elif w > img_w * 0.4 and h > img_h * 0.2:
            return "financial_details"
        else:
            return "paragraph"
    
    def _full_document_fallback(self, image: np.ndarray) -> Dict:
        """Fallback to full document as single region"""
        h, w = image.shape[:2]
        
        region = {
            "bounding_box": {
                "x": 0,
                "y": 0,
                "width": w,
                "height": h
            },
            "region_type": "full_document",
            "confidence": 1.0
        }
        
        print("⚠ Using full document fallback segmentation")
        return {
            "regions": [region],
            "detection_method": "fallback",
            "status": "fallback"
        }