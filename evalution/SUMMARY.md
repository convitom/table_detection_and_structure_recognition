# Table Structure Evaluation System - Summary

## ✅ Đã hoàn thành

Tôi đã tạo hệ thống đánh giá TEDS hoàn chỉnh cho pipeline nhận dạng cấu trúc bảng của bạn.

## 📦 Files đã tạo

### Core System (9 files)
1. **evaluate_teds.py** - Main evaluation script
2. **html_converter.py** - Masks → HTML conversion (FIXED với logic đúng)
3. **teds_metric.py** - TEDS calculation với simple_mode
4. **debug_html_conversion.py** - Debug tool
5. **test_html_converter.py** - Unit tests (ALL PASS ✓)
6. **example_usage.py** - Code examples
7. **requirements.txt** - Dependencies
8. **README.md** - Full documentation
9. **QUICKSTART.md** - Quick start guide

## 🎯 Hiểu biết đã sửa

### Trước (SAI):
- TEDS-Simple = structure only
- TEDS-Complex = structure + content

### Sau (ĐÚNG):
- **TEDS-Simple** = Grid only (rows × cols), KHÔNG có span info
- **TEDS-Complex** = Grid + Spans (rowspan/colspan attributes)
- **Content** = KHÔNG được evaluate trong cả 2 metrics

## 🔧 Logic HTML Conversion (FIXED - HOÀN TOÀN ĐÚNG)

### Critical Understanding:

```
❌ SAI: Xóa separators khỏi grid
✅ ĐÚNG: Track cell merging, giữ nguyên grid

Lý do: Separator có thể cần thiết ở row/col khác!
Ví dụ: col_lines[1] bị "bỏ qua" ở row 0 (span cell)
        NHƯNG vẫn cần ở row 1 (normal cells)
```

### Ví dụ cụ thể:

```
Grid: 2 rows × 4 cols
Row 0: [cell 0-1 MERGED] [cell 2] [cell 3]
Row 1: [cell 0] [cell 1] [cell 2] [cell 3]

Blob: col_span covering row 0, cols 0-1
Best separators: {} (empty)

→ col_lines[1] "không tồn tại" cho row 0 (cells merge)
→ col_lines[1] VẪN tồn tại cho row 1 (cells riêng biệt)

HTML:
Row 0: <td colspan="2"> <td> <td>  → 3 cells
Row 1: <td> <td> <td> <td>        → 4 cells
```

### Algorithm Implementation:

**Step 1: Identify covered cells**
```python
for each blob:
    find all grid cells that overlap with blob bbox
```

**Step 2: Group by row/column**
```python
if col_span:
    group cells by row
    for each row:
        find horizontal merge regions
else:  # row_span
    group cells by column  
    for each column:
        find vertical merge regions
```

**Step 3: Find merge regions**
```python
for each group:
    current_region = [first_cell]
    for each next_cell:
        separator = boundary between current and next
        if separator in best_separators:
            # Best sep → start new region
            save current_region
            current_region = [next_cell]
        else:
            # Non-best sep → continue merging
            current_region.append(next_cell)
```

**Step 4: Mark master and merged cells**
```python
for each merge region with len > 1:
    master = first cell in region
    master_cells[master] = {rowspan/colspan}
    merged_cells.add(other cells in region)
```

**Step 5: Generate HTML**
```python
for each cell in grid:
    if cell in merged_cells:
        skip  # Merged into another cell
    elif cell in master_cells:
        output <td rowspan="X" colspan="Y">
    else:
        output <td>  # Regular cell
```

### Tested & Verified ✓

- Test 1: Partial blob → Only affected rows changed ✓
- Test 2: Best separator → Creates multiple span cells ✓
- Test 3: Original grid preserved outside blobs ✓

## 📊 TEDS Metrics Implementation

### TEDS-Simple (simple_mode=True):
```python
# Remove all rowspan/colspan attributes
# Compare only: <table><tr><td></td>...
# Score = 1.0 nếu số rows và cols giống nhau
```

### TEDS-Complex (simple_mode=False):
```python
# Keep rowspan/colspan attributes
# Compare: <table><tr><td colspan="2">...
# Score phụ thuộc vào cả grid và span structure
```

## 🚀 Sử dụng

### 1. Test HTML converter:
```bash
python test_html_converter.py
# Expected: ALL TESTS PASSED ✓
```

### 2. Debug một bảng:
```bash
python debug_html_conversion.py \
  --data-root /path/to/dataset \
  --gt-jsonl gt.jsonl \
  --table-id 377
```

### 3. Evaluate với GT masks:
```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl gt.jsonl \
  --output-jsonl predictions.jsonl \
  --use-gt-masks
```

**Expected với GT masks**:
- TEDS-Simple: ~100% (nếu grid extraction đúng)
- TEDS-Complex: Phụ thuộc OCR và span logic

### 4. Evaluate với predicted masks:
```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl gt.jsonl \
  --output-jsonl predictions.jsonl \
  --use-pred-masks \
  --row-space-model model.pth \
  --col-space-model model.pth \
  --span-model model.pth \
  --device cuda
```

## ⚠️ Dependencies cần cài

```bash
pip install -r requirements.txt --break-system-packages
```

**Key packages**:
- apted (for TEDS calculation)
- lxml (for HTML parsing)
- opencv-python, numpy, torch
- paddleocr (already in your code)

## 🐛 Debugging Workflow

1. **Chạy tests** → verify logic
2. **Debug 1 table** → understand reconstruction
3. **Evaluate GT masks** → check HTML conversion
4. **Evaluate pred masks** → full pipeline assessment

## 📝 Notes

- Logic đã được test kỹ với unit tests
- HTML converter hiểu đúng best separators logic
- TEDS-Simple và Complex được implement riêng biệt
- Content (text) KHÔNG được evaluate

## 🎓 Tài liệu

Xem:
- **README.md** - Chi tiết đầy đủ
- **QUICKSTART.md** - Bắt đầu nhanh
- **example_usage.py** - Code examples

Chúc may mắn với evaluation! 🚀
