"""
Table Detection Module
Input: Image
Output: List of table bounding boxes
"""
from ultralytics import YOLO
import cv2
import numpy as np


class TableDetector:
    def __init__(self, model_path, conf_threshold=0.1, iou_threshold=0.5):
        """
        Initialize table detector
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, img_or_path):
        """
        Detect tables in image
        
        Args:
            img_or_path: Image array (BGR) or path to image
            
        Returns:
            List of dict with keys: 'bbox' (x1,y1,x2,y2), 'conf', 'class'
        """
        results = self.model.predict(
            source=img_or_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class': cls,
                'class_name': self.model.names[cls]
            })
        
        return detections
    
    def crop_tables(self, img, detections, padding=5):
        """
        Crop detected tables from image
        
        Args:
            img: Original image (BGR or RGB)
            detections: List of detection dicts from detect()
            padding: Padding around bbox
            
        Returns:
            List of cropped table images
        """
        h, w = img.shape[:2]
        cropped_tables = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            cropped = img[y1:y2, x1:x2].copy()
            cropped_tables.append({
                'image': cropped,
                'bbox': (x1, y1, x2, y2),
                'conf': det['conf'],
                'class_name': det['class_name']
            })
        
        return cropped_tables
