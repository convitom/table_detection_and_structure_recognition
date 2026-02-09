"""
Example: Simple usage of Table Pipeline with Configurable Parameters

This example shows how to use the pipeline programmatically
with customizable parameters for fine-tuning.
"""
import cv2
import matplotlib.pyplot as plt
from table_pipeline import TablePipeline


def main():
    # ============= CONFIGURATION =============
    # TODO: Update these paths to your model locations
    TABLE_DETECT_MODEL = r"D:\USTH\Computer_Vision\table\model\table_detect_model.pt"
    ROW_SPACE_MODEL = r"D:\USTH\Computer_Vision\table\model\row_space_seg.pth"
    COL_SPACE_MODEL = r"D:\USTH\Computer_Vision\table\model\col_space_seg.pth"
    SPAN_MODEL = r"D:\USTH\Computer_Vision\table\model\span_seg.pth"
    
    # TODO: Update this to your test image
    IMAGE_PATH = r"D:\USTH\Computer_Vision\table\images_outside\15082021Mutamdandau4.jpg"
    
    # Optional: Save output
    OUTPUT_PATH = r"D:\USTH\Computer_Vision\table\result\result_4.png"  # Set to None to skip saving
    
    # ============= PARAMETER CONFIGURATION =============
    # You can adjust these parameters to fine-tune the pipeline
    # Leave as default (None) or comment out to use built-in defaults
    
    # Table Detection Parameters
    DETECT_CONF_THRESHOLD = 0.01  # Default: 0.1 - Lower = more detections
    DETECT_IOU_THRESHOLD = 0.5    # Default: 0.5 - For NMS
    
    # Space Segmentation Parameters
    SPACE_SEG_THRESHOLD = 0.01     # Default: 0.2 - Binary threshold for row/col spaces
    SPACE_SEG_MIN_RATIO = 0.5    # Default: 0.75 - Min ratio for straightening masks
    
    # Span Segmentation Parameters
    SPAN_SEG_THRESHOLD = 0.5      # Default: 0.5 - Binary threshold for spans
    
    # Table Reconstruction Parameters
    RECONSTRUCT_MIN_LENGTH = 20   # Default: 20 - Min length for line extraction
    RECONSTRUCT_SPAN_THRESHOLD = 0.85  # Default: 0.85 - Threshold for span identification
    GRID_THICKNESS = 1           # Default: 2 - Thickness of grid lines
    
    # ============= INITIALIZE PIPELINE =============
    print("Initializing pipeline...")
    pipeline = TablePipeline(
        table_detect_model_path=TABLE_DETECT_MODEL,
        row_space_model_path=ROW_SPACE_MODEL,
        col_space_model_path=COL_SPACE_MODEL,
        span_model_path=SPAN_MODEL,
        device='cpu',  # or 'cpu'
        use_ocr=True,
        # Configurable parameters
        detect_conf_threshold=DETECT_CONF_THRESHOLD,
        detect_iou_threshold=DETECT_IOU_THRESHOLD,
        space_seg_threshold=SPACE_SEG_THRESHOLD,
        space_seg_min_ratio=SPACE_SEG_MIN_RATIO,
        span_seg_threshold=SPAN_SEG_THRESHOLD,
        reconstruct_min_length=RECONSTRUCT_MIN_LENGTH,
        reconstruct_span_threshold=RECONSTRUCT_SPAN_THRESHOLD,
        grid_thickness=GRID_THICKNESS
    )
    
    # ============= PROCESS IMAGE =============
    print(f"\nProcessing image: {IMAGE_PATH}")
    result = pipeline.process_image(
        IMAGE_PATH,
        visualize=True
    )
    
    # ============= SAVE OUTPUT (OPTIONAL) =============
    if OUTPUT_PATH and result['visualization'] is not None:
        pipeline.save_visualization(OUTPUT_PATH, result['visualization'])
    
    # ============= DISPLAY RESULTS =============
    if result['visualization'] is not None:
        pipeline.display_visualization(result['visualization'], figsize=(20, 15))
    
    # ============= ACCESS INDIVIDUAL RESULTS =============
    print("\n" + "="*50)
    print("DETAILED RESULTS:")
    print("="*50)
    
    for idx, table_result in enumerate(result['results']):
        print(f"\nTable {idx + 1}:")
        print(f"  • Confidence: {table_result['conf']:.3f}")
        print(f"  • Bbox: {table_result['bbox']}")
        
        grid = table_result['structure']['grid']
        row_lines, col_lines = grid
        print(f"  • Grid: {len(row_lines)} rows × {len(col_lines)} columns")
    
    # ============= PARAMETER TUNING GUIDE =============
    print("\n" + "="*50)
    print("PARAMETER TUNING GUIDE:")
    print("="*50)
    print("""
    If you're not getting good results, try adjusting these parameters:
    
    1. DETECT_CONF_THRESHOLD (default: 0.1)
       - Too many false detections? → Increase (e.g., 0.3, 0.5)
       - Missing tables? → Decrease (e.g., 0.05, 0.01)
    
    2. SPACE_SEG_THRESHOLD (default: 0.2)
       - Missing row/column separators? → Decrease (e.g., 0.1)
       - Too many false separators? → Increase (e.g., 0.3, 0.4)
    
    3. SPACE_SEG_MIN_RATIO (default: 0.75)
       - Grid lines incomplete? → Decrease (e.g., 0.5, 0.6)
       - Too many broken/short lines? → Increase (e.g., 0.85, 0.9)
    
    4. SPAN_SEG_THRESHOLD (default: 0.5)
       - Not detecting merged cells? → Decrease (e.g., 0.3, 0.4)
       - Detecting spans where there are none? → Increase (e.g., 0.6, 0.7)
    
    5. RECONSTRUCT_MIN_LENGTH (default: 20)
       - Missing thin columns/rows? → Decrease (e.g., 10, 15)
       - Too many spurious lines? → Increase (e.g., 30, 40)
    
    6. RECONSTRUCT_SPAN_THRESHOLD (default: 0.85)
       - Not identifying span cells? → Decrease (e.g., 0.7, 0.75)
       - Identifying normal cells as spans? → Increase (e.g., 0.9, 0.95)
    
    7. GRID_THICKNESS (default: 2)
       - Grid lines too thick? → Decrease (1)
       - Grid lines hard to see? → Increase (3, 4)
    """)


if __name__ == "__main__":
    main()
