"""
Table Structure Recognition Evaluation using TEDS
Evaluates table structure recognition on test dataset
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

RUN_DIRECT = True
data_root = r"D:\USTH\Computer_Vision\table\crop_table"
row_space_model_path = r"D:\USTH\Computer_Vision\table\model\row_space_seg.pth"
col_space_model_path = r"D:\USTH\Computer_Vision\table\model\col_space_seg.pth"
span_model_path = r"D:\USTH\Computer_Vision\table\model\span_seg.pth"
gt_jsonl = r"D:\USTH\Computer_Vision\table\test.jsonl"
output = r"D:\USTH\Computer_Vision\table\eva3\TEDS_result\predictions.jsonl"
device = 'cpu'
use_gt_masks = False  # Set to False to predict masks instead of using GT masks
SAVE_JSONL_PREDICTIONS = True
SAVE_JSON_TEDS_SUMMARY = True

# ============= EVALUATION PARAMETERS =============
# Space Segmentation parameters (only used when use_gt_masks=False)
SPACE_SEG_THRESHOLD = 0.2
SPACE_SEG_MIN_RATIO = 0.75

# Span Segmentation parameters (only used when use_gt_masks=False)
SPAN_SEG_THRESHOLD = 0.5

# Table Reconstruction parameters (used in both GT and prediction modes)
RECONSTRUCT_MIN_LENGTH = 20
RECONSTRUCT_SPAN_THRESHOLD = 0.85

# Text bbox merge parameters (used in both GT and prediction modes)
COL_SPAN_MERGE_VERTICAL_THRESHOLD = 50
COL_SPAN_MERGE_HORIZONTAL_THRESHOLD = 50
ROW_SPAN_MERGE_VERTICAL_THRESHOLD = 10
ROW_SPAN_MERGE_HORIZONTAL_THRESHOLD = 200

# Add parent directory to path
# sys.path.insert(0, '/mnt/user-data/uploads')
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.space_seg_utils import SpaceSegmentor
from utils.span_seg_utils import SpanSegmentor
from utils.table_reconstruct_utils import TableReconstructor
from html_converter import HTMLConverter
from teds_metric import TEDSMetric


class TableEvaluator:
    def __init__(self, 
                 data_root,
                 row_space_model_path=None,
                 col_space_model_path=None,
                 span_model_path=None,
                 device=None,
                 use_gt_masks=True,
                 # Space Segmentation parameters (only used when use_gt_masks=False)
                 space_seg_threshold=0.2,
                 space_seg_min_ratio=0.75,
                 # Span Segmentation parameters (only used when use_gt_masks=False)
                 span_seg_threshold=0.5,
                 # Table Reconstruction parameters (used in both modes)
                 reconstruct_min_length=20,
                 reconstruct_span_threshold=0.85,
                 # Text bbox merge parameters (used in both modes)
                 col_span_merge_vertical_threshold=5,
                 col_span_merge_horizontal_threshold=50,
                 row_span_merge_vertical_threshold=5,
                 row_span_merge_horizontal_threshold=50):
        """
        Initialize table evaluator
        
        Args:
            data_root: Root directory of dataset
            row_space_model_path: Path to row space model (for prediction)
            col_space_model_path: Path to col space model (for prediction)
            span_model_path: Path to span model (for prediction)
            device: 'cuda' or 'cpu'
            use_gt_masks: If True, use ground truth masks; if False, predict masks
            
            # Space Segmentation parameters (only used when use_gt_masks=False)
            space_seg_threshold: Threshold for binary prediction (default: 0.2)
            space_seg_min_ratio: Min ratio for straightening masks (default: 0.75)
            
            # Span Segmentation parameters (only used when use_gt_masks=False)
            span_seg_threshold: Threshold for binary prediction (default: 0.5)
            
            # Table Reconstruction parameters (used in both modes)
            reconstruct_min_length: Min length for line extraction (default: 20)
            reconstruct_span_threshold: Threshold for span region identification (default: 0.85)
            
            # Text bbox merge parameters (used in both modes)
            col_span_merge_vertical_threshold: Max vertical gap for merging text boxes in col span cells (default: 5)
            col_span_merge_horizontal_threshold: Max horizontal distance between centers for col span cells (default: 50)
            row_span_merge_vertical_threshold: Max vertical gap for merging text boxes in row span cells (default: 5)
            row_span_merge_horizontal_threshold: Max horizontal distance between centers for row span cells (default: 50)
        """
        self.data_root = Path(data_root)
        self.use_gt_masks = use_gt_masks
        
        # Store parameters
        self.space_seg_threshold = space_seg_threshold
        self.space_seg_min_ratio = space_seg_min_ratio
        self.span_seg_threshold = span_seg_threshold
        self.reconstruct_min_length = reconstruct_min_length
        self.reconstruct_span_threshold = reconstruct_span_threshold
        
        # Initialize components
        self.reconstructor = TableReconstructor(
            use_ocr=True,
            col_span_merge_vertical_threshold=col_span_merge_vertical_threshold,
            col_span_merge_horizontal_threshold=col_span_merge_horizontal_threshold,
            row_span_merge_vertical_threshold=row_span_merge_vertical_threshold,
            row_span_merge_horizontal_threshold=row_span_merge_horizontal_threshold
        )
        self.html_converter = HTMLConverter()
        
        # Initialize segmentors if predicting masks
        if not use_gt_masks:
            if not all([row_space_model_path, col_space_model_path, span_model_path]):
                raise ValueError("Model paths required when use_gt_masks=False")
            
            print(f"  ├─ Loading space segmentor...")
            self.space_segmentor = SpaceSegmentor(
                row_space_model_path,
                col_space_model_path,
                device=device
            )
            print(f"  └─ Loading span segmentor...")
            self.span_segmentor = SpanSegmentor(
                span_model_path,
                device=device
            )
        
        # Initialize TEDS metrics
        self.teds_metric = TEDSMetric(structure_only=False)
    
    def load_ground_truth(self, gt_jsonl_path):
        gt_data = {}
        bad_lines = 0

        with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    table_id = data['table_id']
                    gt_data[table_id] = data
                except json.JSONDecodeError as e:
                    bad_lines += 1
                    print(
                        f"[WARN] Skip invalid JSON at line {idx}: {e}"
                    )
                except KeyError:
                    bad_lines += 1
                    print(
                        f"[WARN] Missing table_id at line {idx}"
                    )

        print(f"Loaded {len(gt_data)} GT tables")
        if bad_lines > 0:
            print(f"[WARN] Skipped {bad_lines} invalid lines in GT file")

        return gt_data

    
    def _load_mask(self, mask_path):
        """Load mask from file"""
        if not os.path.exists(mask_path):
            return None
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask
    
    def _load_image(self, image_path):
        """Load image from file"""
        if not os.path.exists(image_path):
            return None
        
        img = cv2.imread(str(image_path))
        return img
    
    def evaluate_single(self, table_id, gt_data):
        """
        Evaluate a single table
        
        Args:
            table_id: Table ID (e.g., 1, 2, 3...)
            gt_data: Ground truth data for this table
        
        Returns:
            dict: Evaluation results
        """
        # Construct file paths
        table_name = f"table_{table_id}.png"
        image_path = self.data_root / "images" / "test" / table_name
        
        # Check if image exists
        if not image_path.exists():
            return {
                'table_id': table_id,
                'status': 'missing_image',
                'teds_simple': 0.0,
                'teds_complex': 0.0
            }
        
        # Load image
        img = self._load_image(image_path)
        if img is None:
            return {
                'table_id': table_id,
                'status': 'failed_load_image',
                'teds_simple': 0.0,
                'teds_complex': 0.0
            }
        
        # Load or predict masks
        if self.use_gt_masks:
            # Load GT masks
            row_space_path = self.data_root / "row_space_masks" / "test" / table_name
            col_space_path = self.data_root / "col_space_masks" / "test" / table_name
            row_span_path = self.data_root / "row_span_masks" / "test" / table_name
            col_span_path = self.data_root / "col_span_masks" / "test" / table_name
            
            row_space_mask = self._load_mask(row_space_path)
            col_space_mask = self._load_mask(col_space_path)
            row_span_mask = self._load_mask(row_span_path)
            col_span_mask = self._load_mask(col_span_path)
            
            # Check if all masks exist
            if any(m is None for m in [row_space_mask, col_space_mask, row_span_mask, col_span_mask]):
                return {
                    'table_id': table_id,
                    'status': 'missing_masks',
                    'teds_simple': 0.0,
                    'teds_complex': 0.0
                }
        else:
            # Predict masks
            try:
                # Convert BGR to RGB for segmentors
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Predict space masks
                row_space_mask, col_space_mask = self.space_segmentor.segment(
                    img_rgb,
                    threshold=self.space_seg_threshold
                )
                
                # Clean masks with configurable min_ratio
                row_space_mask = self.space_segmentor._straighten_row_mask(
                    row_space_mask,
                    min_ratio=self.space_seg_min_ratio
                )
                col_space_mask = self.space_segmentor._straighten_col_mask(
                    col_space_mask,
                    min_ratio=self.space_seg_min_ratio
                )
                
                # Predict span masks
                row_span_mask, col_span_mask = self.span_segmentor.segment(
                    img_rgb, 
                    row_space_mask, 
                    col_space_mask,
                    threshold=self.span_seg_threshold
                )
                
                # Convert to 0-255 range
                row_space_mask = (row_space_mask * 255).astype(np.uint8)
                col_space_mask = (col_space_mask * 255).astype(np.uint8)
                row_span_mask = (row_span_mask * 255).astype(np.uint8)
                col_span_mask = (col_span_mask * 255).astype(np.uint8)
                
            except Exception as e:
                return {
                    'table_id': table_id,
                    'status': f'prediction_failed: {str(e)}',
                    'teds_simple': 0.0,
                    'teds_complex': 0.0
                }
        
        # Reconstruct table structure with configurable parameters
        try:
            result = self.reconstructor.reconstruct(
                img, 
                row_space_mask, 
                col_space_mask,
                row_span_mask, 
                col_span_mask,
                min_length=self.reconstruct_min_length,
                span_threshold=self.reconstruct_span_threshold
            )
        except Exception as e:
            return {
                'table_id': table_id,
                'status': f'reconstruction_failed: {str(e)}',
                'teds_simple': 0.0,
                'teds_complex': 0.0
            }
        
        # Convert to HTML
        try:
            pred_html = self.html_converter.convert_to_html(result)
        except Exception as e:
            return {
                'table_id': table_id,
                'status': f'html_conversion_failed: {str(e)}',
                'teds_simple': 0.0,
                'teds_complex': 0.0
            }
        
        # Get ground truth HTML
        gt_html = gt_data['html']
        
        # Compute TEDS scores
        try:
            teds_simple = self.teds_metric.compute_teds(pred_html, gt_html, simple_mode=True)
            teds_complex = self.teds_metric.compute_teds(pred_html, gt_html, simple_mode=False)
        except Exception as e:
            return {
                'table_id': table_id,
                'status': f'teds_computation_failed: {str(e)}',
                'teds_simple': 0.0,
                'teds_complex': 0.0,
                'pred_html': pred_html,
                'gt_html': gt_html
            }
        
        return {
            'table_id': table_id,
            'status': 'success',
            'teds_simple': teds_simple,
            'teds_complex': teds_complex,
            'pred_html': pred_html,
            'gt_html': gt_html
        }
    
    def evaluate_dataset(self, gt_jsonl_path):
        print(f"Loading ground truth from {gt_jsonl_path}...")
        gt_data = self.load_ground_truth(gt_jsonl_path)
        print(f"Loaded {len(gt_data)} ground truth entries")

        results = []
        jsonl_objects = []

        print(f"\nEvaluating tables...")
        for table_id, gt in tqdm(gt_data.items(), desc="Processing"):
            result = self.evaluate_single(table_id, gt)
            results.append(result)

            # ---- JSONL: log EVERYTHING ----
            jsonl_objects.append({
                'table_id': table_id,
                'status': result['status'],
                'teds_simple': result.get('teds_simple'),
                'teds_complex': result.get('teds_complex'),
                'pred_html': result.get('pred_html')
            })

        # ---------- STRICT FILTER ----------
        successful = [
            r for r in results
            if r.get('status') == 'success'
            and r.get('teds_simple') is not None
            and r.get('teds_complex') is not None
        ]

        # ---------- NO SUCCESS CASE ----------
        if len(successful) == 0:
            stats = {
                'total': len(results),
                'successful': 0,
                'failed': len(results),
                'teds_simple': None,
                'teds_complex': None
            }
            return stats, jsonl_objects

        # ---------- COMPUTE STATS ----------
        teds_simple_scores = np.array([r['teds_simple'] for r in successful])
        teds_complex_scores = np.array([r['teds_complex'] for r in successful])

        stats = {
            'total': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'teds_simple': {
                'mean': float(teds_simple_scores.mean()),
                'median': float(np.median(teds_simple_scores)),
                'std': float(teds_simple_scores.std()),
                'min': float(teds_simple_scores.min()),
                'max': float(teds_simple_scores.max())
            },
            'teds_complex': {
                'mean': float(teds_complex_scores.mean()),
                'median': float(np.median(teds_complex_scores)),
                'std': float(teds_complex_scores.std()),
                'min': float(teds_complex_scores.min()),
                'max': float(teds_complex_scores.max())
            }
        }

        return stats, jsonl_objects
    
    def print_summary(self, stats):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total tables: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['teds_simple'] is not None:
            print()
            print("TEDS-Simple (Structure Only):")
            print(f"  Mean:   {stats['teds_simple']['mean']:.4f}")
            print(f"  Median: {stats['teds_simple']['median']:.4f}")
            print(f"  Std:    {stats['teds_simple']['std']:.4f}")
            print(f"  Min:    {stats['teds_simple']['min']:.4f}")
            print(f"  Max:    {stats['teds_simple']['max']:.4f}")
            print()
            print("TEDS-Complex (With Content):")
            print(f"  Mean:   {stats['teds_complex']['mean']:.4f}")
            print(f"  Median: {stats['teds_complex']['median']:.4f}")
            print(f"  Std:    {stats['teds_complex']['std']:.4f}")
            print(f"  Min:    {stats['teds_complex']['min']:.4f}")
            print(f"  Max:    {stats['teds_complex']['max']:.4f}")
        else:
            print("\n[ERROR] No successful evaluations - cannot compute TEDS metrics")
        
        print("="*60)


def main():
    if RUN_DIRECT:
        args = argparse.Namespace(
            data_root=data_root,
            row_space_model=row_space_model_path,
            col_space_model=col_space_model_path,
            span_model=span_model_path,
            gt_jsonl=gt_jsonl,
            output_jsonl=output,
            use_gt_masks=use_gt_masks,
            device=device,
            # Space Segmentation parameters
            space_seg_threshold=SPACE_SEG_THRESHOLD,
            space_seg_min_ratio=SPACE_SEG_MIN_RATIO,
            # Span Segmentation parameters
            span_seg_threshold=SPAN_SEG_THRESHOLD,
            # Table Reconstruction parameters
            reconstruct_min_length=RECONSTRUCT_MIN_LENGTH,
            reconstruct_span_threshold=RECONSTRUCT_SPAN_THRESHOLD,
            # Text bbox merge parameters
            col_span_merge_vertical_threshold=COL_SPAN_MERGE_VERTICAL_THRESHOLD,
            col_span_merge_horizontal_threshold=COL_SPAN_MERGE_HORIZONTAL_THRESHOLD,
            row_span_merge_vertical_threshold=ROW_SPAN_MERGE_VERTICAL_THRESHOLD,
            row_span_merge_horizontal_threshold=ROW_SPAN_MERGE_HORIZONTAL_THRESHOLD
        )
    else:
        parser = argparse.ArgumentParser(description="Evaluate table structure recognition using TEDS")
        
        parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset')
        parser.add_argument('--gt-jsonl', type=str, required=True,
                        help='Path to ground truth JSONL file')
        parser.add_argument('--output-jsonl', type=str, default=None,
                        help='Path to save prediction JSONL')
        parser.add_argument('--use-gt-masks', action='store_true',
                        help='Use ground truth masks (for debugging)')
        parser.add_argument('--use-pred-masks', action='store_true',
                        help='Predict masks from models')
        
        # Model paths (required if --use-pred-masks)
        parser.add_argument('--row-space-model', type=str, default=None,
                        help='Path to row space segmentation model')
        parser.add_argument('--col-space-model', type=str, default=None,
                        help='Path to column space segmentation model')
        parser.add_argument('--span-model', type=str, default=None,
                        help='Path to span segmentation model')
        parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use')
        
        # Space Segmentation parameters (only used when use_gt_masks=False)
        parser.add_argument('--space-seg-threshold', type=float, default=0.2,
                        help='Threshold for binary prediction in space segmentation (default: 0.2)')
        parser.add_argument('--space-seg-min-ratio', type=float, default=0.75,
                        help='Min ratio for straightening row/col masks (default: 0.75)')
        
        # Span Segmentation parameters (only used when use_gt_masks=False)
        parser.add_argument('--span-seg-threshold', type=float, default=0.5,
                        help='Threshold for binary prediction in span segmentation (default: 0.5)')
        
        # Table Reconstruction parameters (used in both modes)
        parser.add_argument('--reconstruct-min-length', type=int, default=20,
                        help='Min length for line extraction in reconstruction (default: 20)')
        parser.add_argument('--reconstruct-span-threshold', type=float, default=0.85,
                        help='Threshold for span region identification (default: 0.85)')
        
        # Text bbox merge parameters (used in both modes)
        parser.add_argument('--col-span-merge-vertical-threshold', type=int, default=5,
                        help='Max vertical gap for merging text boxes in col span cells (default: 5)')
        parser.add_argument('--col-span-merge-horizontal-threshold', type=int, default=50,
                        help='Max horizontal distance between centers for col span cells (default: 50)')
        parser.add_argument('--row-span-merge-vertical-threshold', type=int, default=5,
                        help='Max vertical gap for merging text boxes in row span cells (default: 5)')
        parser.add_argument('--row-span-merge-horizontal-threshold', type=int, default=50,
                        help='Max horizontal distance between centers for row span cells (default: 50)')
        
        args = parser.parse_args()
    
    # Print evaluation configuration
    print("="*60)
    print("EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Mode: {'GT Masks' if args.use_gt_masks else 'Predicted Masks'}")
    print(f"Device: {args.device}")
    print()
    
    if not args.use_gt_masks:
        print("Space Segmentation Parameters:")
        print(f"  • Threshold: {args.space_seg_threshold}")
        print(f"  • Min Ratio: {args.space_seg_min_ratio}")
        print()
        print("Span Segmentation Parameters:")
        print(f"  • Threshold: {args.span_seg_threshold}")
        print()
    
    print("Table Reconstruction Parameters:")
    print(f"  • Min Length: {args.reconstruct_min_length}")
    print(f"  • Span Threshold: {args.reconstruct_span_threshold}")
    print()
    print("Text Bbox Merge Parameters:")
    print(f"  • Col Span Vertical Threshold: {args.col_span_merge_vertical_threshold}")
    print(f"  • Col Span Horizontal Threshold: {args.col_span_merge_horizontal_threshold}")
    print(f"  • Row Span Vertical Threshold: {args.row_span_merge_vertical_threshold}")
    print(f"  • Row Span Horizontal Threshold: {args.row_span_merge_horizontal_threshold}")
    print("="*60)
    print()
    
    # Initialize evaluator
    print("🔧 Initializing Evaluator...")
    evaluator = TableEvaluator(
        data_root=args.data_root,
        row_space_model_path=args.row_space_model,
        col_space_model_path=args.col_space_model,
        span_model_path=args.span_model,
        device=args.device,
        use_gt_masks=args.use_gt_masks,
        # Space Segmentation parameters
        space_seg_threshold=args.space_seg_threshold,
        space_seg_min_ratio=args.space_seg_min_ratio,
        # Span Segmentation parameters
        span_seg_threshold=args.span_seg_threshold,
        # Table Reconstruction parameters
        reconstruct_min_length=args.reconstruct_min_length,
        reconstruct_span_threshold=args.reconstruct_span_threshold,
        # Text bbox merge parameters
        col_span_merge_vertical_threshold=args.col_span_merge_vertical_threshold,
        col_span_merge_horizontal_threshold=args.col_span_merge_horizontal_threshold,
        row_span_merge_vertical_threshold=args.row_span_merge_vertical_threshold,
        row_span_merge_horizontal_threshold=args.row_span_merge_horizontal_threshold
    )
    print("✅ Evaluator initialized!\n")
    
    # Run evaluation
    stats, jsonl_objects = evaluator.evaluate_dataset(
        gt_jsonl_path=args.gt_jsonl
    )
    
    # Print summary
    evaluator.print_summary(stats)
    
    # Save detailed results
    if SAVE_JSONL_PREDICTIONS and args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving JSONL predictions to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for obj in jsonl_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    
    if SAVE_JSON_TEDS_SUMMARY and args.output_jsonl:
        summary_path = Path(args.output_jsonl).with_name(
            Path(args.output_jsonl).stem + '_results.json'
        )

        print(f"💾 Saving TEDS summary to {summary_path}")
        
        # Add evaluation configuration to summary
        stats['evaluation_config'] = {
            'use_gt_masks': args.use_gt_masks,
            'space_seg_threshold': args.space_seg_threshold,
            'space_seg_min_ratio': args.space_seg_min_ratio,
            'span_seg_threshold': args.span_seg_threshold,
            'reconstruct_min_length': args.reconstruct_min_length,
            'reconstruct_span_threshold': args.reconstruct_span_threshold,
            'col_span_merge_vertical_threshold': args.col_span_merge_vertical_threshold,
            'col_span_merge_horizontal_threshold': args.col_span_merge_horizontal_threshold,
            'row_span_merge_vertical_threshold': args.row_span_merge_vertical_threshold,
            'row_span_merge_horizontal_threshold': args.row_span_merge_horizontal_threshold
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
