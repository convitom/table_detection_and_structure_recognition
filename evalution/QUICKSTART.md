# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt --break-system-packages
```

## Quick Test

### 1. Test HTML Converter Logic

```bash
python test_html_converter.py
```

Expected output: All tests passed ✓

### 2. Debug Single Table (requires your dataset)

```bash
python debug_html_conversion.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/gt.jsonl \
  --table-id 377
```

Outputs:
- `debug_table_377_grid.png` - Grid visualization
- `debug_table_377_lines.png` - Table with lines
- `debug_table_377_analysis.json` - Detailed analysis

### 3. Evaluate with GT Masks (for debugging conversion logic)

```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/gt.jsonl \
  --output-jsonl predictions_gt.jsonl \
  --use-gt-masks
```

**Expected**: TEDS-Simple should be close to 100% if HTML conversion is correct

### 4. Evaluate with Predicted Masks (full pipeline)

```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/gt.jsonl \
  --output-jsonl predictions_pred.jsonl \
  --use-pred-masks \
  --row-space-model models/row_space.pth \
  --col-space-model models/col_space.pth \
  --span-model models/span.pth \
  --device cuda
```

## Understanding Results

### TEDS-Simple (Structure-only)
- Compares only table structure (grid + spans)
- Should be ~100% with GT masks
- Lower with predicted masks indicates segmentation errors

### TEDS-Complex (With content)
- Compares structure AND text content
- Will be lower because OCR is needed for text
- Currently only detects text bboxes, doesn't extract content

## Troubleshooting

### TEDS-Simple < 100% with GT masks?
→ Bug in HTML conversion logic
→ Run `debug_html_conversion.py` to inspect

### TEDS-Complex very low?
→ Text content not being extracted
→ Need to add OCR recognition (not just detection)

### Memory errors?
→ Use `--device cpu`
→ Process smaller batches

### Missing images/masks?
→ Check file paths and naming
→ Script skips missing files automatically

## Files Overview

- `evaluate_teds.py` - Main evaluation script
- `html_converter.py` - Masks → HTML conversion
- `teds_metric.py` - TEDS calculation
- `debug_html_conversion.py` - Debug tool
- `test_html_converter.py` - Unit tests
- `example_usage.py` - Code examples

## Next Steps

1. Run tests to verify setup
2. Debug a few tables to understand the pipeline
3. Evaluate with GT masks to check conversion logic
4. Evaluate with predicted masks for full assessment
