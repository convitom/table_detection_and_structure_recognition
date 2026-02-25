"""
Generate row/col span masks for FinTabNet dataset by combining:
- HTML structure (colspan/rowspan info)
- Space masks (actual spacing between rows/cols)
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import cv2


class SpanMaskGenerator:
    def __init__(self, jsonl_path: str, image_dir: str, 
                 row_space_dir: str, col_space_dir: str,
                 output_row_span_dir: str, output_col_span_dir: str):
        """
        Args:
            jsonl_path: Path to FinTabNet JSONL file
            image_dir: Directory containing table images
            row_space_dir: Directory containing row space masks
            col_space_dir: Directory containing column space masks
            output_row_span_dir: Output directory for row span masks
            output_col_span_dir: Output directory for col span masks
        """
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.row_space_dir = Path(row_space_dir)
        self.col_space_dir = Path(col_space_dir)
        self.output_row_span_dir = Path(output_row_span_dir)
        self.output_col_span_dir = Path(output_col_span_dir)
        
        # Create output directories
        self.output_row_span_dir.mkdir(parents=True, exist_ok=True)
        self.output_col_span_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_html_structure(self, html_tokens: List[str]) -> List[List[Dict]]:
        grid = []
        current_row = []
        in_cell = False
        colspan = 1
        rowspan = 1
        is_header = False

        i = 0
        while i < len(html_tokens):
            token = html_tokens[i]

            if token == '<tr>':
                current_row = []

            elif token == '</tr>':
                if current_row:
                    grid.append(current_row)

            elif token == '<td>' or token == '<th>':
                in_cell = True
                colspan = 1
                rowspan = 1
                is_header = (token == '<th>')

            elif token == '<td' or token == '<th':
                in_cell = True
                is_header = (token == '<th')
                colspan = 1
                rowspan = 1

                j = i + 1
                while j < len(html_tokens) and html_tokens[j] not in ['>', '<td>', '<th>']:
                    current_token = html_tokens[j]

                    if 'colspan=' in current_token:
                        try:
                            num_str = current_token.split('colspan=')[1].strip('"\'>')
                            colspan = int(num_str)
                        except:
                            pass

                    if 'rowspan=' in current_token:
                        try:
                            num_str = current_token.split('rowspan=')[1].strip('"\'>')
                            rowspan = int(num_str)
                        except:
                            pass

                    j += 1

                i = j
                continue

            elif token == '</td>' or token == '</th>':
                if in_cell:
                    cell = {
                        'colspan': colspan,
                        'rowspan': rowspan,
                        'is_header': is_header,
                        'is_merged': False
                    }
                    current_row.append(cell)
                    in_cell = False

            i += 1

        # ⚠️ CHỈ RETURN Ở CUỐI
        filled_grid = self._fill_merged_cells(grid)
        return filled_grid

    
    def _fill_merged_cells(self, grid: List[List[Dict]]) -> List[List[Dict]]:
        """
        Fill grid with placeholders for merged cells.
        """
        if not grid:
            return grid
        
        # Determine grid dimensions
        max_cols = max(len(row) for row in grid) if grid else 0
        max_rows = len(grid)
        
        # Create filled grid
        filled_grid = [[None for _ in range(max_cols * 10)] for _ in range(max_rows)]
        
        for row_idx, row in enumerate(grid):
            col_idx = 0
            for cell in row:
                # Find next available column
                while col_idx < len(filled_grid[row_idx]) and filled_grid[row_idx][col_idx] is not None:
                    col_idx += 1
                
                # Place the cell
                filled_grid[row_idx][col_idx] = cell
                
                # Fill colspan
                for c in range(1, cell['colspan']):
                    if col_idx + c < len(filled_grid[row_idx]):
                        filled_grid[row_idx][col_idx + c] = {
                            'colspan': 0,
                            'rowspan': 0,
                            'is_header': cell['is_header'],
                            'is_merged': True,
                            'master_cell': (row_idx, col_idx)
                        }
                
                # Fill rowspan
                for r in range(1, cell['rowspan']):
                    if row_idx + r < len(filled_grid):
                        for c in range(cell['colspan']):
                            if col_idx + c < len(filled_grid[row_idx + r]):
                                filled_grid[row_idx + r][col_idx + c] = {
                                    'colspan': 0,
                                    'rowspan': 0,
                                    'is_header': cell['is_header'],
                                    'is_merged': True,
                                    'master_cell': (row_idx, col_idx)
                                }
                
                col_idx += 1
        
        # Trim empty columns
        max_col = 0
        for row in filled_grid:
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    max_col = max(max_col, col_idx)
        
        filled_grid = [row[:max_col + 1] for row in filled_grid]
        
        return filled_grid
    
    def extract_grid_lines(self, space_mask: np.ndarray, axis: int) -> List[int]:
        """
        Extract grid line positions from space mask.
        
        Args:
            space_mask: Binary mask (white=space, black=content)
            axis: 0 for rows (horizontal lines), 1 for cols (vertical lines)
        
        Returns:
            List of line positions (pixel coordinates)
        """
        # For row lines: sum along width (axis=1) to get vertical projection
        # For col lines: sum along height (axis=0) to get horizontal projection
        projection = np.sum(space_mask, axis=axis)
        
        # Normalize to 0-1 range
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        
        # White regions (value close to 1) are spaces between rows/cols
        # Find peaks where projection > threshold
        threshold = 0.7  # Adjust this if needed
        is_space = projection > threshold
        
        # Find the center of each white space region
        lines = []
        in_space = False
        space_start = 0
        
        for i in range(len(is_space)):
            if is_space[i] and not in_space:
                # Start of a space region
                space_start = i
                in_space = True
            elif not is_space[i] and in_space:
                # End of space region, record center point
                space_center = (space_start + i - 1) // 2
                lines.append(space_center)
                in_space = False
        
        # Handle case where space extends to the end
        if in_space:
            space_center = (space_start + len(is_space) - 1) // 2
            lines.append(space_center)
        
        return lines
    
    def create_span_mask(self, image_shape: Tuple[int, int], 
                        grid: List[List[Dict]], 
                        row_lines: List[int], 
                        col_lines: List[int],
                        span_type: str) -> np.ndarray:
        """
        Create row or col span mask.
        
        Args:
            image_shape: (height, width)
            grid: Parsed HTML grid structure
            row_lines: Y coordinates of row boundaries
            col_lines: X coordinates of column boundaries
            span_type: 'row' or 'col'
        
        Returns:
            Binary mask where white indicates span regions
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add boundaries
        row_boundaries = [0] + row_lines + [height]
        col_boundaries = [0] + col_lines + [width]
        
        for row_idx, row in enumerate(grid):
            if row_idx >= len(row_boundaries) - 1:
                break
            
            for col_idx, cell in enumerate(row):
                if col_idx >= len(col_boundaries) - 1:
                    break
                
                if cell is None:
                    continue
                
                # Check if this cell has span
                has_span = False
                if span_type == 'row' and cell['rowspan'] > 1:
                    has_span = True
                elif span_type == 'col' and cell['colspan'] > 1:
                    has_span = True
                
                if has_span:
                    # Calculate cell boundaries
                    y1 = row_boundaries[row_idx]
                    y2 = row_boundaries[min(row_idx + cell['rowspan'], len(row_boundaries) - 1)]
                    x1 = col_boundaries[col_idx]
                    x2 = col_boundaries[min(col_idx + cell['colspan'], len(col_boundaries) - 1)]
                    
                    # Fill the span region (including spaces)
                    mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def process_table(self, table_data: Dict, debug: bool = False) -> Tuple[bool, str]:
        """
        Process a single table and generate span masks.
        
        Args:
            debug: If True, save debug visualizations
        
        Returns:
            (success, message)
        """
        table_id = table_data['table_id']
        image_name = f"table_{table_id}.png"
        
        # Check if all required files exist
        image_path = self.image_dir / image_name
        row_space_path = self.row_space_dir / image_name
        col_space_path = self.col_space_dir / image_name
        
        if not image_path.exists():
            return False, f"Image not found: {image_name}"
        if not row_space_path.exists():
            return False, f"Row space mask not found: {image_name}"
        if not col_space_path.exists():
            return False, f"Col space mask not found: {image_name}"
        
        # Load image and masks
        image = Image.open(image_path)
        row_space_mask = np.array(Image.open(row_space_path).convert('L'))
        col_space_mask = np.array(Image.open(col_space_path).convert('L'))
        
        # Parse HTML structure
        html_tokens = table_data['html']['structure']['tokens']
        grid = self.parse_html_structure(html_tokens)
        
        # Extract grid lines from space masks
        row_lines = self.extract_grid_lines(row_space_mask, axis=1)
        col_lines = self.extract_grid_lines(col_space_mask, axis=0)
        
        # Debug visualization
        if debug:
            self._save_debug_visualization(
                image, row_space_mask, col_space_mask,
                row_lines, col_lines, grid, image_name
            )
        
        # Create span masks
        row_span_mask = self.create_span_mask(
            image.size[::-1], grid, row_lines, col_lines, 'row'
        )
        col_span_mask = self.create_span_mask(
            image.size[::-1], grid, row_lines, col_lines, 'col'
        )
        
        # Save masks
        row_span_path = self.output_row_span_dir / image_name
        col_span_path = self.output_col_span_dir / image_name
        
        Image.fromarray(row_span_mask).save(row_span_path)
        Image.fromarray(col_span_mask).save(col_span_path)
        
        return True, f"Successfully processed {image_name}"
    
    def _save_debug_visualization(self, image, row_space_mask, col_space_mask,
                                   row_lines, col_lines, grid, image_name):
        """Save debug visualization showing detected grid lines."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with grid lines
        axes[0, 0].imshow(image)
        for y in row_lines:
            axes[0, 0].axhline(y, color='red', linewidth=1, alpha=0.7)
        for x in col_lines:
            axes[0, 0].axvline(x, color='blue', linewidth=1, alpha=0.7)
        axes[0, 0].set_title(f'Original + Grid Lines\n{len(row_lines)} rows, {len(col_lines)} cols')
        axes[0, 0].axis('off')
        
        # Row space mask
        axes[0, 1].imshow(row_space_mask, cmap='gray')
        for y in row_lines:
            axes[0, 1].axhline(y, color='red', linewidth=1)
        axes[0, 1].set_title('Row Space Mask + Detected Lines')
        axes[0, 1].axis('off')
        
        # Col space mask
        axes[1, 0].imshow(col_space_mask, cmap='gray')
        for x in col_lines:
            axes[1, 0].axvline(x, color='blue', linewidth=1)
        axes[1, 0].set_title('Col Space Mask + Detected Lines')
        axes[1, 0].axis('off')
        
        # Grid structure
        axes[1, 1].text(0.1, 0.9, f'Grid structure: {len(grid)} rows', 
                       transform=axes[1, 1].transAxes, fontsize=10)
        
        grid_info = []
        for i, row in enumerate(grid[:10]):  # Show first 10 rows
            row_info = []
            for j, cell in enumerate(row[:10]):  # Show first 10 cols
                if cell is None:
                    row_info.append('None')
                elif cell.get('is_merged'):
                    row_info.append('M')
                else:
                    rs = cell.get('rowspan', 1)
                    cs = cell.get('colspan', 1)
                    if rs > 1 or cs > 1:
                        row_info.append(f'{rs}x{cs}')
                    else:
                        row_info.append('1x1')
            grid_info.append(f"Row {i}: {' | '.join(row_info)}")
        
        axes[1, 1].text(0.1, 0.8, '\n'.join(grid_info[:15]), 
                       transform=axes[1, 1].transAxes, 
                       fontsize=8, verticalalignment='top', family='monospace')
        axes[1, 1].set_title('HTML Grid Structure')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        debug_path = self.output_row_span_dir.parent / f'debug_{image_name}'
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved debug visualization: {debug_path}")
    
    def process_all(self, debug_first_n: int = 0):
        """
        Process all tables in the JSONL file.
        
        Args:
            debug_first_n: If > 0, save debug visualizations for first N tables
        """
        print(f"Reading tables from: {self.jsonl_path}")
        
        success_count = 0
        fail_count = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    table_data = json.loads(line.strip())
                    debug = (debug_first_n > 0 and success_count < debug_first_n)
                    success, message = self.process_table(table_data, debug=debug)
                    
                    if success:
                        success_count += 1
                        if success_count % 100 == 0:
                            print(f"Processed {success_count} tables...")
                    else:
                        fail_count += 1
                        print(f"Line {line_num}: {message}")
                        
                except Exception as e:
                    fail_count += 1
                    print(f"Line {line_num}: Error - {str(e)}")
                    import traceback
                    if debug_first_n > 0 and success_count < 5:
                        traceback.print_exc()
        
        print(f"\n=== Summary ===")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Row span masks saved to: {self.output_row_span_dir}")
        print(f"Col span masks saved to: {self.output_col_span_dir}")

def main():
    """
    Auto-run version: không cần truyền đường dẫn khi chạy file.
    Script sẽ tự lấy path tương đối theo vị trí file .py
    """

    # Lấy thư mục chứa file .py hiện tại
    

    # ==== TÙY CHỈNH Ở ĐÂY NẾU CẦN ====
    x = "val"
    jsonl_path = r"D:\USTH\Computer_Vision\table\test.jsonl"
    image_dir = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
    row_space_dir = r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\test"
    col_space_dir = r"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\test"

    output_row_span_dir = r"D:\USTH\Computer_Vision\table\crop_table\row_span_masks\test"
    output_col_span_dir = r"D:\USTH\Computer_Vision\table\crop_table\col_span_masks\test"
    # =================================

    print("=== Auto configuration ===")
    print("JSONL:", jsonl_path)
    print("Images:", image_dir)
    print("Row space:", row_space_dir)
    print("Col space:", col_space_dir)
    print("Output row span:", output_row_span_dir)
    print("Output col span:", output_col_span_dir)
    print("==========================")

    generator = SpanMaskGenerator(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        row_space_dir=row_space_dir,
        col_space_dir=col_space_dir,
        output_row_span_dir=output_row_span_dir,
        output_col_span_dir=output_col_span_dir
    )

    generator.process_all()



if __name__ == '__main__':
    main()
