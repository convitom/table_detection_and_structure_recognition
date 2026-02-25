# Table Structure Recognition Evaluation with TEDS

Hệ thống đánh giá cấu trúc bảng sử dụng TEDS (Tree Edit Distance based Similarity) metric.

## Tổng quan

Hệ thống này đánh giá chất lượng của việc nhận dạng cấu trúc bảng bằng cách:

1. **Chuyển đổi từ 4 masks sang HTML structure**: Từ row/col space masks và row/col span masks → HTML tokens
2. **Tính toán TEDS score**: So sánh cây HTML predicted với ground truth
3. **Hỗ trợ 2 chế độ đánh giá**:
   - **TEDS-Simple (Grid only)**: Chỉ so sánh grid structure (số hàng, số cột) - loại bỏ rowspan/colspan
   - **TEDS-Complex (Grid + Spans)**: So sánh cả grid và các merged cells (rowspan/colspan attributes)

## Cấu trúc Dataset

```
data_root/
├── images/
│   └── test/
│       ├── table_1.png
│       ├── table_2.png
│       └── ...
├── row_space_masks/
│   └── test/
│       ├── table_1.png
│       └── ...
├── col_space_masks/
│   └── test/
│       ├── table_1.png
│       └── ...
├── row_span_masks/
│   └── test/
│       ├── table_1.png
│       └── ...
└── col_span_masks/
    └── test/
        ├── table_1.png
        └── ...
```

Ground truth JSONL format:
```json
{
  "table_id": 377,
  "html": {
    "structure": {
      "tokens": ["<table>", "<tr>", "<td>", "</td>", ...]
    }
  },
  "split": "test",
  "filename": "test/table_377.png"
}
```

## Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt --break-system-packages
```

## Sử dụng

### 1. Đánh giá với Ground Truth Masks (Debugging)

Sử dụng GT masks để debug logic chuyển đổi HTML:

```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/ground_truth.jsonl \
  --output-jsonl predictions_gt.jsonl \
  --use-gt-masks
```

**Kết quả mong đợi**: 
- TEDS-Simple nên đạt ~100% nếu logic chuyển đổi HTML đúng
- TEDS-Complex có thể thấp hơn do OCR detection sai

### 2. Đánh giá với Predicted Masks (Full Pipeline)

Sử dụng models để predict masks:

```bash
python evaluate_teds.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/ground_truth.jsonl \
  --output-jsonl predictions_pred.jsonl \
  --use-pred-masks \
  --row-space-model /path/to/row_space_model.pth \
  --col-space-model /path/to/col_space_model.pth \
  --span-model /path/to/span_model.pth \
  --device cuda
```

### 3. Debug một bảng cụ thể

Phân tích chi tiết một bảng để tìm lỗi:

```bash
python debug_html_conversion.py \
  --data-root /path/to/dataset \
  --gt-jsonl /path/to/ground_truth.jsonl \
  --table-id 377
```

Outputs:
- `debug_table_377_grid.png`: Visualization của grid và span blobs
- `debug_table_377_lines.png`: Bảng với grid lines
- `debug_table_377_analysis.json`: Chi tiết phân tích

## Giải thích Logic

### 1. Space Masks → Grid Lines

- **Row space mask**: Mask tô khoảng trống giữa 2 hàng (màu trắng là border)
- **Col space mask**: Mask tô khoảng trống giữa 2 cột

Từ masks này, extract các đường separator lines → tạo grid (như bàn cờ caro)

Ví dụ: 9 row lines + 9 col lines → grid 8x8 cells

### 2. Span Masks → Span Regions

- **Row span mask**: Vùng của row span cells (cells merge nhiều hàng)
- **Col span mask**: Vùng của col span cells (cells merge nhiều cột)

**Vấn đề**: Mask prediction thường tô nối liền nhiều span cells → blob

**Giải pháp**: Sử dụng OCR text detection để tách blob thành các span cells riêng biệt

### 3. OCR Text Detection → Best Separators

**Quy trình**:

1. Từ span mask, xác định cells nào là span cell (>85% diện tích cell được tô)
2. Tìm connected components → các blobs (vùng span liền kề)
3. Trên mỗi blob:
   - Detect text bboxes bằng PaddleOCR
   - Tìm các "best separators" = separators KHÔNG cắt qua text
   - Best separators CHIA blob thành nhiều span cells riêng biệt
   - Separators KHÔNG phải best separator → cells được merge qua separator đó

**Ví dụ cụ thể** (nhìn vào result_5.png):
- Header row "2017" có col_span blob covering 2 columns
- Separator giữa "Shares" và "Weighted-Average" KHÔNG phải best separator
- → "2017" là 1 span cell với colspan=2
- Separator này sẽ có gaps khi vẽ visualization

### 4. Grid + Best Separators → HTML Structure

**Thuật toán**:

```python
# Step 1: Build effective grid
effective_grid = original_grid - non_best_separators

# Step 2: For each cell in effective grid
for each cell in effective_grid:
    # Count how many original cells this effective cell covers
    rowspan = count_original_rows_covered
    colspan = count_original_cols_covered
    
    # Output HTML
    if rowspan > 1 or colspan > 1:
        output <td rowspan="..." colspan="...">
    else:
        output <td>
```

**Ví dụ cụ thể**:

```
Original grid: 
- row_lines = [0, 100, 200]  → 2 rows
- col_lines = [0, 100, 200, 300]  → 3 cols
- Total: 2×3 = 6 cells

Blob: col_span covering (0,0) to (200,100)
- Covers cols 0-1 (2 columns)
- Internal separator: col_lines[1] = 100
- Best separators: {} (empty)

Effective grid:
- col_lines[1] được XÓA
- effective_col_lines = [0, 200, 300]  → 2 cols
- Total: 2×2 = 4 cells

Cell spans:
- Cell (0,0) in effective grid covers 2 original cols → colspan=2
- All others: rowspan=1, colspan=1

HTML:
Row 0: <td colspan="2"> <td>
Row 1: <td> <td>
```

**Key insight**: 
- Best separators = separators GIỮ LẠI trong effective grid
- Non-best separators = separators BỊ XÓA khỏi effective grid
- Rowspan/colspan = số lượng original cells được merge

## Phân tích Vấn đề

### Tại sao TEDS-Simple không đạt 100% với GT masks?

TEDS-Simple chỉ so sánh grid structure (số rows, số cols), không quan tâm spans.

Có thể do:

1. **Grid extraction sai**:
   - Miss một số separator lines từ space masks
   - Extract thừa separator lines
   - Boundary lines (top/bottom/left/right) không được thêm đúng

2. **Giải pháp**: Debug với `debug_html_conversion.py` để xem grid có đúng không

### Tại sao TEDS-Complex thấp hơn TEDS-Simple?

TEDS-Complex so sánh cả grid + spans (rowspan/colspan).

Có thể do:

1. **OCR detection sai**:
   - Text detection miss một số vùng
   - Detect nhiều text boxes cho 1 span cell
   - Best separator selection sai

2. **Logic chuyển đổi HTML sai**:
   - Cách xác định master cell sai
   - Cách xử lý best separators sai
   - Rowspan/colspan calculation sai

3. **Giải pháp**: 
   - Debug với một bảng đơn giản
   - Kiểm tra best_separators có hợp lý không
   - So sánh HTML tokens với GT

### Lưu ý

- TEDS-Simple nên đạt ~100% nếu grid extraction đúng
- TEDS-Complex phụ thuộc nhiều vào OCR và logic xử lý spans
- Content (text recognition) KHÔNG được đánh giá trong cả 2 metrics này

## Debugging Workflow

1. **Chạy với GT masks** → kiểm tra TEDS-Simple
   ```bash
   python evaluate_teds.py --use-gt-masks ...
   ```

2. **Nếu TEDS-Simple < 100%** → debug HTML conversion:
   ```bash
   python debug_html_conversion.py --table-id X
   ```
   
   Kiểm tra:
   - Grid có đúng số rows/cols không?
   - Span blobs có đúng không?
   - Best separators có hợp lý không?
   - HTML tokens có match với GT không?

3. **Fix bugs** trong:
   - `html_converter.py`: Logic chuyển đổi HTML
   - `table_reconstruct_utils.py`: Logic tìm span cells và separators

4. **Chạy lại** cho đến khi TEDS-Simple ≈ 100%

5. **Chạy với predicted masks** → đánh giá model
   ```bash
   python evaluate_teds.py --use-pred-masks ...
   ```

## Files

- `evaluate_teds.py`: Main evaluation script
- `html_converter.py`: Convert reconstruction result → HTML structure
- `teds_metric.py`: TEDS metric calculation
- `debug_html_conversion.py`: Debug tool cho single table
- `table_reconstruct_utils.py`: Table reconstruction (đã có)
- `space_seg_utils.py`: Space segmentation (đã có)
- `span_seg_utils.py`: Span segmentation (đã có)

## Output Format

### Prediction JSONL
```json
{
  "table_id": 377,
  "html": {
    "structure": {
      "tokens": ["<table>", "<tr>", "<td>", ...]
    }
  }
}
```

### Results JSON
```json
{
  "total": 100,
  "successful": 95,
  "failed": 5,
  "teds_simple": {
    "mean": 0.9234,
    "median": 0.9567,
    "std": 0.0823,
    "min": 0.5432,
    "max": 1.0000
  },
  "teds_complex": {
    "mean": 0.8123,
    ...
  }
}
```

## Khắc phục Vấn đề Thường Gặp

### 1. TEDS-Simple thấp với GT masks

**Nguyên nhân**: Logic HTML conversion sai

**Giải pháp**:
- Debug với `debug_html_conversion.py`
- Kiểm tra `html_converter.py` → fix logic `_build_span_map()`
- Test lại với một vài bảng đơn giản

### 2. OCR không detect text

**Nguyên nhân**: PaddleOCR config sai hoặc image quality thấp

**Giải pháp**:
- Kiểm tra PaddleOCR setup
- Thử tăng resolution của crop images
- Xem log của OCR trong debug mode

### 3. Best separator selection sai

**Nguyên nhân**: Logic trong `_find_best_separators_combination()` chưa tối ưu

**Giải pháp**:
- Cải thiện thuật toán tìm separator
- Xem xét thêm constraints (khoảng cách, alignment...)

### 4. Memory issues với dataset lớn

**Giải pháp**:
- Process theo batch nhỏ
- Sử dụng `--device cpu` nếu cần
- Clear cache định kỳ

## Contact & Support

Nếu gặp vấn đề:
1. Chạy debug script để phân tích chi tiết
2. Kiểm tra log output
3. Xem visualization để hiểu vấn đề
