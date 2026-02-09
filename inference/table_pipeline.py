"""
End-to-End Table Detection and Structure Recognition Pipeline

Usage:
    python table_pipeline.py --image path/to/image.png --output path/to/output.png

This pipeline:
1. Detects tables in the image
2. For each detected table:
   - Segments row and column spaces
   - Segments row and column spans
   - Reconstructs table structure
3. Draws bboxes and grid lines on the original image
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from table_detect_utils import TableDetector
from space_seg_utils import SpaceSegmentor
from span_seg_utils import SpanSegmentor
from table_reconstruct_utils import TableReconstructor


class TablePipeline:
    def __init__(self, 
                 table_detect_model_path,
                 row_space_model_path,
                 col_space_model_path,
                 span_model_path,
                 device=None,
                 use_ocr=True,
                 # Table Detection parameters
                 detect_conf_threshold=0.1,
                 detect_iou_threshold=0.5,
                 # Space Segmentation parameters
                 space_seg_threshold=0.2,
                 space_seg_min_ratio=0.75,
                 # Span Segmentation parameters
                 span_seg_threshold=0.5,
                 # Table Reconstruction parameters
                 reconstruct_min_length=20,
                 reconstruct_span_threshold=0.85,
                 grid_thickness=2):
        """
        Initialize end-to-end table pipeline
        
        Args:
            table_detect_model_path: Path to YOLO table detection model
            row_space_model_path: Path to row space segmentation model
            col_space_model_path: Path to column space segmentation model
            span_model_path: Path to span segmentation model
            device: 'cuda' or 'cpu'
            use_ocr: Whether to use OCR in reconstruction
            
            # Table Detection parameters
            detect_conf_threshold: Confidence threshold for table detection (default: 0.1)
            detect_iou_threshold: IoU threshold for NMS (default: 0.5)
            
            # Space Segmentation parameters
            space_seg_threshold: Threshold for binary prediction in space segmentation (default: 0.2)
            space_seg_min_ratio: Min ratio for straightening masks (default: 0.75)
            
            # Span Segmentation parameters
            span_seg_threshold: Threshold for binary prediction in span segmentation (default: 0.5)
            
            # Table Reconstruction parameters
            reconstruct_min_length: Min length for line extraction (default: 20)
            reconstruct_span_threshold: Threshold for span region identification (default: 0.85)
            grid_thickness: Thickness of grid lines in visualization (default: 2)
        """
        print("🔧 Initializing Table Pipeline...")
        
        # Store parameters
        self.detect_conf_threshold = detect_conf_threshold
        self.detect_iou_threshold = detect_iou_threshold
        self.space_seg_threshold = space_seg_threshold
        self.space_seg_min_ratio = space_seg_min_ratio
        self.span_seg_threshold = span_seg_threshold
        self.reconstruct_min_length = reconstruct_min_length
        self.reconstruct_span_threshold = reconstruct_span_threshold
        self.grid_thickness = grid_thickness
        
        # Initialize components
        print("  ├─ Loading table detector...")
        self.detector = TableDetector(
            table_detect_model_path,
            conf_threshold=detect_conf_threshold,
            iou_threshold=detect_iou_threshold
        )
        
        print("  ├─ Loading space segmentor...")
        self.space_segmentor = SpaceSegmentor(
            row_space_model_path, 
            col_space_model_path, 
            device=device
        )
        
        print("  ├─ Loading span segmentor...")
        self.span_segmentor = SpanSegmentor(
            span_model_path, 
            device=device
        )
        
        print("  └─ Loading table reconstructor...")
        self.reconstructor = TableReconstructor(use_ocr=use_ocr)
        
        print("✅ Pipeline initialized successfully!\n")
    
    def process_image(self, img_path, conf_threshold=None, visualize=True):
        """
        Process a single image through the full pipeline
        
        Args:
            img_path: Path to input image
            conf_threshold: Confidence threshold for table detection (uses init value if None)
            visualize: Whether to return visualization
            
        Returns:
            dict with keys:
                - 'image': original image
                - 'detections': list of table detections
                - 'results': list of reconstruction results for each table
                - 'visualization': image with bbox and grids drawn (if visualize=True)
        """
        print(f"📄 Processing image: {img_path}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Detect tables
        print("  [1/4] 🔍 Detecting tables...")
        if conf_threshold is not None:
            self.detector.conf_threshold = conf_threshold
        detections = self.detector.detect(img)
        print(f"        Found {len(detections)} table(s)")
        
        if len(detections) == 0:
            print("  ⚠️  No tables detected!")
            return {
                'image': img_rgb,
                'detections': [],
                'results': [],
                'visualization': img_rgb.copy() if visualize else None
            }
        
        # Crop tables
        cropped_tables = self.detector.crop_tables(img_rgb, detections)
        
        # Process each table
        results = []
        for idx, table_data in enumerate(cropped_tables):
            print(f"\n  📊 Processing Table {idx + 1}/{len(cropped_tables)}...")
            table_img = table_data['image']
            
            # Step 2: Space segmentation
            print(f"     [2/4] 🧩 Segmenting spaces...")
            row_space_mask, col_space_mask = self.space_segmentor.segment(
                table_img,
                threshold=self.space_seg_threshold
            )
            
            # Apply straightening with configurable min_ratio
            row_space_mask = self.space_segmentor._straighten_row_mask(
                row_space_mask, 
                min_ratio=self.space_seg_min_ratio
            )
            col_space_mask = self.space_segmentor._straighten_col_mask(
                col_space_mask,
                min_ratio=self.space_seg_min_ratio
            )
            
            # Step 3: Span segmentation
            print(f"     [3/4] 🔗 Segmenting spans...")
            row_span_mask, col_span_mask = self.span_segmentor.segment(
                table_img, 
                row_space_mask, 
                col_space_mask,
                threshold=self.span_seg_threshold
            )
            
            # Step 4: Reconstruct structure
            print(f"     [4/4] 🏗️  Reconstructing structure...")
            # Convert to BGR for OCR (PaddleOCR expects BGR)
            table_img_bgr = cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR)
            
            result = self.reconstructor.reconstruct(
                table_img_bgr,
                row_space_mask * 255,  # Convert to 0-255
                col_space_mask * 255,
                row_span_mask * 255,
                col_span_mask * 255,
                min_length=self.reconstruct_min_length,
                span_threshold=self.reconstruct_span_threshold
            )
            
            # Draw grid on table
            table_with_grid = self.reconstructor.draw_grid(
                table_img, 
                result,
                color=(255, 0, 0),  # Red in RGB
                thickness=self.grid_thickness
            )
            
            results.append({
                'table_idx': idx,
                'bbox': table_data['bbox'],
                'conf': table_data['conf'],
                'table_image': table_img,
                'row_space_mask': row_space_mask,
                'col_space_mask': col_space_mask,
                'row_span_mask': row_span_mask,
                'col_span_mask': col_span_mask,
                'structure': result,
                'table_with_grid': table_with_grid
            })
        
        print("\n✅ Processing complete!")
        
        # Create visualization
        vis = None
        if visualize:
            vis = self._create_visualization(img_rgb, detections, results)
        
        return {
            'image': img_rgb,
            'detections': detections,
            'results': results,
            'visualization': vis
        }
    
    def _create_visualization(self, img, detections, results):
        """
        Create visualization with bboxes and grids
        
        Args:
            img: Original image (RGB)
            detections: List of detection dicts
            results: List of processing results
            
        Returns:
            Image with bboxes and grids drawn
        """
        vis = img.copy()
        
        # Draw each table
        for det, res in zip(detections, results):
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(
                vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            
            # Overlay table with grid
            table_with_grid = res['table_with_grid']
            
            # Resize grid to fit bbox
            h, w = y2 - y1, x2 - x1
            grid_resized = cv2.resize(table_with_grid, (w, h))
            
            # Blend with original
            vis[y1:y2, x1:x2] = grid_resized
        
        return vis
    
    def save_visualization(self, output_path, visualization):
        """Save visualization to file"""
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), vis_bgr)
        print(f"💾 Saved visualization to: {output_path}")
    
    def display_visualization(self, visualization, figsize=(15, 10)):
        """Display visualization using matplotlib"""
        plt.figure(figsize=figsize)
        plt.imshow(visualization)
        plt.axis('off')
        plt.title("Table Detection and Structure Recognition")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Table Detection and Structure Recognition"
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output', 
        type=str,
        default=None,
        help='Path to save output visualization (optional)'
    )
    parser.add_argument(
        '--table-detect-model',
        type=str,
        required=True,
        help='Path to YOLO table detection model'
    )
    parser.add_argument(
        '--row-space-model',
        type=str,
        required=True,
        help='Path to row space segmentation model'
    )
    parser.add_argument(
        '--col-space-model',
        type=str,
        required=True,
        help='Path to column space segmentation model'
    )
    parser.add_argument(
        '--span-model',
        type=str,
        required=True,
        help='Path to span segmentation model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (cuda or cpu, auto-detect if not specified)'
    )
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR in reconstruction'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display visualization'
    )
    
    # Table Detection parameters
    parser.add_argument(
        '--detect-conf-threshold',
        type=float,
        default=0.1,
        help='Confidence threshold for table detection (default: 0.1)'
    )
    parser.add_argument(
        '--detect-iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for NMS in table detection (default: 0.5)'
    )
    
    # Space Segmentation parameters
    parser.add_argument(
        '--space-seg-threshold',
        type=float,
        default=0.2,
        help='Threshold for binary prediction in space segmentation (default: 0.2)'
    )
    parser.add_argument(
        '--space-seg-min-ratio',
        type=float,
        default=0.75,
        help='Min ratio for straightening row/col masks (default: 0.75)'
    )
    
    # Span Segmentation parameters
    parser.add_argument(
        '--span-seg-threshold',
        type=float,
        default=0.5,
        help='Threshold for binary prediction in span segmentation (default: 0.5)'
    )
    
    # Table Reconstruction parameters
    parser.add_argument(
        '--reconstruct-min-length',
        type=int,
        default=20,
        help='Min length for line extraction in reconstruction (default: 20)'
    )
    parser.add_argument(
        '--reconstruct-span-threshold',
        type=float,
        default=0.85,
        help='Threshold for span region identification (default: 0.85)'
    )
    parser.add_argument(
        '--grid-thickness',
        type=int,
        default=2,
        help='Thickness of grid lines in visualization (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TablePipeline(
        table_detect_model_path=args.table_detect_model,
        row_space_model_path=args.row_space_model,
        col_space_model_path=args.col_space_model,
        span_model_path=args.span_model,
        device=args.device,
        use_ocr=not args.no_ocr,
        # Pass all configurable parameters
        detect_conf_threshold=args.detect_conf_threshold,
        detect_iou_threshold=args.detect_iou_threshold,
        space_seg_threshold=args.space_seg_threshold,
        space_seg_min_ratio=args.space_seg_min_ratio,
        span_seg_threshold=args.span_seg_threshold,
        reconstruct_min_length=args.reconstruct_min_length,
        reconstruct_span_threshold=args.reconstruct_span_threshold,
        grid_thickness=args.grid_thickness
    )
    
    # Process image
    result = pipeline.process_image(
        args.image,
        visualize=True
    )
    
    # Save output if specified
    if args.output:
        pipeline.save_visualization(args.output, result['visualization'])
    
    # Display if not disabled
    if not args.no_display and result['visualization'] is not None:
        pipeline.display_visualization(result['visualization'])
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  • Tables detected: {len(result['detections'])}")
    for idx, res in enumerate(result['results']):
        grid_info = res['structure']['grid']
        print(f"  • Table {idx + 1}: {len(grid_info[0])} rows × {len(grid_info[1])} columns")


if __name__ == "__main__":
    main()
