# End-to-End Table Detection and Structure Recognition Pipeline

Integrated pipeline for detecting and reconstructing table structures from images with configurable parameters.

## 📋 Overview

This pipeline combines 4 processing steps:

1. **Table Detection**: Detect table positions in images (YOLO)
2. **Space Segmentation**: Segment empty spaces between rows/columns (U-Net)
3. **Span Segmentation**: Identify merged cells (U-Net)
4. **Table Reconstruction**: Reconstruct table grid structure (OCR + logic)

### Input
- Image containing one or more tables

### Output
- Image with bounding boxes around detected tables
- Grid lines drawn on each table
- Detailed structure information for each table

## 🗂️ File Structure

```
.
├── table_detect_utils.py       # Table detection module
├── space_seg_utils.py          # Space segmentation module
├── span_seg_utils.py           # Span segmentation module
├── table_reconstruct_utils.py  # Table reconstruction module
├── table_pipeline.py           # Main pipeline (CLI)
├── example_usage.py            # Simple usage example
└── README.md                   # This file
```

## 🚀 Installation

### Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install ultralytics
pip install opencv-python
pip install paddlepaddle paddleocr
pip install matplotlib numpy
```

### Required Models

You need 4 models:
1. `table_detect_model.pt` - YOLO model for table detection
2. `row_space_seg.pth` - U-Net model for row space
3. `col_space_seg.pth` - U-Net model for column space
4. `span_seg.pth` - U-Net model for span cells
5. paddleOCR (already called in pipeline)
## ⚙️ Configurable Parameters

This updated version allows you to easily configure all important parameters across the pipeline modules.

### 1. Table Detection Parameters

**`detect_conf_threshold`** (default: 0.1)
- **What it does**: Minimum confidence score for a detection to be considered a table
- **When to adjust**:
  - Increase (0.3-0.5) if you're getting too many false positive detections
  - Decrease (0.01-0.05) if tables are being missed

**`detect_iou_threshold`** (default: 0.5)
- **What it does**: IoU threshold for Non-Maximum Suppression to remove duplicate detections
- **When to adjust**:
  - Increase (0.6-0.7) if you want to keep more overlapping detections
  - Decrease (0.3-0.4) if you're getting duplicate boxes for the same table

### 2. Space Segmentation Parameters

**`space_seg_threshold`** (default: 0.2)
- **What it does**: Binary threshold to convert probability predictions to masks for row/column spaces
- **When to adjust**:
  - Decrease (0.1-0.15) if row/column separators are being missed
  - Increase (0.3-0.5) if you're getting too many false separator predictions

**`space_seg_min_ratio`** (default: 0.75)
- **What it does**: Minimum ratio of row width (for rows) or column height (for columns) that must be filled to consider a line valid
- **When to adjust**:
  - Decrease (0.5-0.6) if grid lines are incomplete or broken
  - Increase (0.85-0.95) if you want stricter filtering of separator lines

### 3. Span Segmentation Parameters

**`span_seg_threshold`** (default: 0.5)
- **What it does**: Binary threshold to detect merged cells (row spans and column spans)
- **When to adjust**:
  - Decrease (0.3-0.4) if merged cells are not being detected
  - Increase (0.6-0.7) if normal cells are being incorrectly identified as merged cells

### 4. Table Reconstruction Parameters

**`reconstruct_min_length`** (default: 20)
- **What it does**: Minimum pixel length for a line to be considered a valid row/column separator
- **When to adjust**:
  - Decrease (10-15) if you have very narrow columns or rows that are being missed
  - Increase (30-50) if you're getting spurious short lines

**`reconstruct_span_threshold`** (default: 0.85)
- **What it does**: Threshold for determining if a cell is a span cell (85% of cell must be covered)
- **When to adjust**:
  - Decrease (0.6-0.75) if merged cells aren't being identified
  - Increase (0.9-0.95) if normal cells are being marked as spans

**`grid_thickness`** (default: 2)
- **What it does**: Thickness of grid lines in the output visualization
- **When to adjust**:
  - Decrease (1) for thinner, more subtle lines
  - Increase (3-5) for thicker, more visible lines

## 📖 Usage

### Method 1: Command Line Interface

**Basic usage:**
```bash
python table_pipeline.py \
    --image path/to/your/image.png \
    --output path/to/output.png \
    --table-detect-model path/to/table_detect_model.pt \
    --row-space-model path/to/table_row_space_seg.pth \
    --col-space-model path/to/table_col_space_seg_2.pth \
    --span-model path/to/span_seg_2.pth \
    --device cuda
```

**With custom parameters:**
```bash
python table_pipeline.py \
    --image path/to/image.png \
    --table-detect-model path/to/table_detect.pt \
    --row-space-model path/to/row_space.pth \
    --col-space-model path/to/col_space.pth \
    --span-model path/to/span.pth \
    --detect-conf-threshold 0.05 \
    --space-seg-threshold 0.15 \
    --space-seg-min-ratio 0.6 \
    --span-seg-threshold 0.4 \
    --reconstruct-min-length 15 \
    --reconstruct-span-threshold 0.75 \
    --grid-thickness 3
```

**Available Arguments:**
- `--image`: Path to input image (required)
- `--output`: Path to save result image (optional)
- `--table-detect-model`: Path to YOLO model (required)
- `--row-space-model`: Path to row space model (required)
- `--col-space-model`: Path to column space model (required)
- `--span-model`: Path to span model (required)
- `--device`: Device to use - `cuda` or `cpu` (auto-detect if not specified)
- `--no-ocr`: Disable OCR in reconstruction
- `--no-display`: Don't display results
- `--detect-conf-threshold`: Confidence threshold for detection (default: 0.1)
- `--detect-iou-threshold`: IoU threshold for NMS (default: 0.5)
- `--space-seg-threshold`: Threshold for space segmentation (default: 0.2)
- `--space-seg-min-ratio`: Min ratio for straightening masks (default: 0.75)
- `--span-seg-threshold`: Threshold for span segmentation (default: 0.5)
- `--reconstruct-min-length`: Min length for line extraction (default: 20)
- `--reconstruct-span-threshold`: Threshold for span identification (default: 0.85)
- `--grid-thickness`: Thickness of grid lines (default: 2)

### Method 2: Python API

**Basic usage:**
```python
from table_pipeline import TablePipeline

# Initialize pipeline with default parameters
pipeline = TablePipeline(
    table_detect_model_path="path/to/table_detect_model.pt",
    row_space_model_path="path/to/table_row_space_seg.pth",
    col_space_model_path="path/to/table_col_space_seg_2.pth",
    span_model_path="path/to/span_seg_2.pth",
    device='cuda',  # or 'cpu'
    use_ocr=True
)

# Process image
result = pipeline.process_image(
    "path/to/image.png",
    visualize=True
)

# Save output
if result['visualization'] is not None:
    pipeline.save_visualization("output.png", result['visualization'])

# Display
pipeline.display_visualization(result['visualization'])
```

**With custom parameters:**
```python
from table_pipeline import TablePipeline

pipeline = TablePipeline(
    table_detect_model_path="path/to/model.pt",
    row_space_model_path="path/to/row.pth",
    col_space_model_path="path/to/col.pth",
    span_model_path="path/to/span.pth",
    device='cpu',
    use_ocr=True,
    # Custom parameters
    detect_conf_threshold=0.05,
    detect_iou_threshold=0.5,
    space_seg_threshold=0.15,
    space_seg_min_ratio=0.6,
    span_seg_threshold=0.4,
    reconstruct_min_length=15,
    reconstruct_span_threshold=0.75,
    grid_thickness=3
)

result = pipeline.process_image("path/to/image.png")

# Access detailed results
for idx, table in enumerate(result['results']):
    print(f"Table {idx + 1}:")
    print(f"  Confidence: {table['conf']:.3f}")
    print(f"  Bbox: {table['bbox']}")
    
    grid = table['structure']['grid']
    print(f"  Grid: {len(grid[0])} rows × {len(grid[1])} columns")
```

### Method 3: Using Example Script

1. Open `example_usage.py`
2. Update model and image paths:
   ```python
   TABLE_DETECT_MODEL = r"path/to/your/table_detect_model.pt"
   ROW_SPACE_MODEL = r"path/to/your/table_row_space_seg.pth"
   COL_SPACE_MODEL = r"path/to/your/table_col_space_seg_2.pth"
   SPAN_MODEL = r"path/to/your/span_seg_2.pth"
   IMAGE_PATH = r"path/to/your/image.png"
   ```
3. Adjust parameters as needed (all parameters are clearly documented in the file)
4. Run:
   ```bash
   python example_usage.py
   ```

## 📊 Output Format

### Result Dictionary

```python
{
    'image': np.array,              # Original image (RGB)
    'detections': [                 # List of detected tables
        {
            'bbox': (x1, y1, x2, y2),
            'conf': float,
            'class': int,
            'class_name': str
        },
        ...
    ],
    'results': [                    # Processing results for each table
        {
            'table_idx': int,
            'bbox': (x1, y1, x2, y2),
            'conf': float,
            'table_image': np.array,        # Cropped table image
            'row_space_mask': np.array,     # Row space mask
            'col_space_mask': np.array,     # Column space mask
            'row_span_mask': np.array,      # Row span mask
            'col_span_mask': np.array,      # Column span mask
            'structure': {                   # Table structure
                'grid': (row_lines, col_lines),
                'span_blobs': [...]
            },
            'table_with_grid': np.array     # Table image with grid drawn
        },
        ...
    ],
    'visualization': np.array       # Final image with bboxes and grids
}
```

## 🎨 Visualization

The pipeline creates a visualization with:
- **Green bounding boxes**: Around each detected table
- **Confidence scores**: Displayed above each bbox
- **Red grid lines**: Table structure grid
  - Solid lines: Normal separators
  - Lines with gaps: Inside span cells (only best separators drawn)

## 🔧 Advanced Usage

### Processing Batch Images

```python
import glob
from pathlib import Path

image_paths = glob.glob("images/*.png")
for img_path in image_paths:
    result = pipeline.process_image(img_path)
    output_path = f"outputs/{Path(img_path).stem}_result.png"
    pipeline.save_visualization(output_path, result['visualization'])
```

### Disable OCR (if not needed)

```python
pipeline = TablePipeline(
    ...,
    use_ocr=False  # Faster but may be less accurate
)
```

### Custom Visualization

```python
# Get results without automatic visualization
result = pipeline.process_image(image_path, visualize=False)

# Create your own visualization
for table_result in result['results']:
    table_with_grid = table_result['table_with_grid']
    # Your custom visualization code here
```

## 🔍 Troubleshooting

### Common Issues

| Problem | Parameter to Adjust | Direction |
|---------|-------------------|-----------|
| Missing tables | `detect_conf_threshold` | Decrease |
| False table detections | `detect_conf_threshold` | Increase |
| Incomplete grid lines | `space_seg_min_ratio` | Decrease |
| Missing row/col separators | `space_seg_threshold` | Decrease |
| Missing merged cells | `span_seg_threshold` | Decrease |
| False span detections | `span_seg_threshold` | Increase |
| Missing thin columns | `reconstruct_min_length` | Decrease |
| Too many spurious lines | `reconstruct_min_length` | Increase |

### CUDA Out of Memory
```python
# Use CPU instead
pipeline = TablePipeline(..., device='cpu')
```

### PaddleOCR Errors
```bash
# Reinstall PaddleOCR
pip uninstall paddleocr paddlepaddle
pip install paddlepaddle-gpu  # or paddlepaddle for CPU
pip install paddleocr
```

### Import Errors
```bash
# Ensure all utility files are in the same directory as table_pipeline.py
```

## 📝 Notes

- The pipeline automatically converts between RGB/BGR when needed
- OCR (PaddleOCR) is only used to detect text positions, not to recognize text
- Grid lines may have gaps in span cells to avoid cutting through text
- Each table is processed independently, results are drawn on the original image
- All parameters use sensible defaults that work for most cases
- Start by adjusting one parameter at a time
- Test on representative samples of your data
- Some parameters interact with each other, so you may need to adjust multiple parameters


