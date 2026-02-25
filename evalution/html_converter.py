"""
HTML Converter Module - CORRECT Implementation

KEY UNDERSTANDING:
- Grid (row_lines, col_lines) = định nghĩa TẤT CẢ cells có thể có
- Span blobs = vùng chứa merged cells
- Best separators = separators CHIA blob thành nhiều merged regions
- Non-best separators = separators bị "ignored" (cells merge qua chúng)

CRITICAL: Không thể xóa separators khỏi grid vì separator có thể cần thiết
ở row/col khác! Thay vào đó, phải track cell merging.

ALGORITHM:
1. Start với original grid (all cells independent)
2. For each blob, xác định region nào merge với nhau dựa trên best separators
3. Mark merged cells và xác định master cell + rowspan/colspan
4. Generate HTML: skip merged cells, output master cells với spans
"""
import numpy as np


class HTMLConverter:
    def __init__(self):
        """Initialize HTML converter"""
        pass
    
    def _build_cell_merge_info(self, row_lines, col_lines, span_blobs):
        """
        Determine which cells merge together based on span blobs and best separators
        
        Returns:
            master_cells: dict {(r, c): {'rowspan': int, 'colspan': int}}
            merged_cells: set of (r, c) that are merged into other cells
        """
        num_rows = len(row_lines) - 1
        num_cols = len(col_lines) - 1
        
        master_cells = {}
        merged_cells = set()
        
        for blob in span_blobs:
            bx1, by1, bx2, by2 = blob['bbox']
            blob_type = blob['type']
            best_seps = blob['best_separators']
            
            # Find which grid cells are covered by this blob
            covered_cells = []
            for r in range(num_rows):
                y1, y2 = row_lines[r], row_lines[r + 1]
                for c in range(num_cols):
                    x1, x2 = col_lines[c], col_lines[c + 1]
                    
                    # Cell is covered if it overlaps with blob
                    if not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2):
                        covered_cells.append((r, c))
            
            if not covered_cells:
                continue
            
            if blob_type == 'col_span':
                # Group cells by row, then find horizontal merges
                rows_dict = {}
                for r, c in covered_cells:
                    if r not in rows_dict:
                        rows_dict[r] = []
                    rows_dict[r].append(c)
                
                for r, cols in rows_dict.items():
                    cols = sorted(cols)
                    
                    # Find merge regions within this row
                    # Cells merge together if NO best separator between them
                    merge_regions = []
                    current_region = [cols[0]]
                    
                    for i in range(1, len(cols)):
                        prev_col = cols[i-1]
                        curr_col = cols[i]
                        
                        # Check if there's a best separator between them
                        # Separator is at col_lines[curr_col]
                        separator_pos = col_lines[curr_col]
                        
                        if separator_pos in best_seps:
                            # Best separator -> start new region
                            merge_regions.append(current_region)
                            current_region = [curr_col]
                        else:
                            # No best separator -> continue merging
                            current_region.append(curr_col)
                    
                    merge_regions.append(current_region)
                    
                    # Create merged cells for each region
                    for region in merge_regions:
                        if len(region) > 1:
                            # This is a merged cell
                            master = (r, region[0])
                            colspan = len(region)
                            
                            if master not in merged_cells:
                                master_cells[master] = {'rowspan': 1, 'colspan': colspan}
                                
                                # Mark other cells as merged
                                for c in region[1:]:
                                    merged_cells.add((r, c))
            
            else:  # row_span
                # Group cells by column, then find vertical merges
                cols_dict = {}
                for r, c in covered_cells:
                    if c not in cols_dict:
                        cols_dict[c] = []
                    cols_dict[c].append(r)
                
                for c, rows in cols_dict.items():
                    rows = sorted(rows)
                    
                    # Find merge regions within this column
                    merge_regions = []
                    current_region = [rows[0]]
                    
                    for i in range(1, len(rows)):
                        prev_row = rows[i-1]
                        curr_row = rows[i]
                        
                        # Separator is at row_lines[curr_row]
                        separator_pos = row_lines[curr_row]
                        
                        if separator_pos in best_seps:
                            merge_regions.append(current_region)
                            current_region = [curr_row]
                        else:
                            current_region.append(curr_row)
                    
                    merge_regions.append(current_region)
                    
                    # Create merged cells
                    for region in merge_regions:
                        if len(region) > 1:
                            master = (region[0], c)
                            rowspan = len(region)
                            
                            if master not in merged_cells:
                                master_cells[master] = {'rowspan': rowspan, 'colspan': 1}
                                
                                for r in region[1:]:
                                    merged_cells.add((r, c))
        
        return master_cells, merged_cells
    
    def convert_to_html(self, reconstruction_result):
        """
        Convert reconstruction result to HTML structure
        
        Args:
            reconstruction_result: Dict from TableReconstructor.reconstruct()
                - 'grid': (row_lines, col_lines)
                - 'span_blobs': list of span blob info
        
        Returns:
            dict: HTML structure
        """
        row_lines, col_lines = reconstruction_result['grid']
        span_blobs = reconstruction_result['span_blobs']
        
        num_rows = len(row_lines) - 1
        num_cols = len(col_lines) - 1
        
        # Build merge info
        master_cells, merged_cells = self._build_cell_merge_info(
            row_lines, col_lines, span_blobs
        )
        
        # Generate HTML tokens
        tokens = ['<table>']
        
        for r in range(num_rows):
            tokens.append('<tr>')
            
            for c in range(num_cols):
                # Skip if this cell is merged into another
                if (r, c) in merged_cells:
                    continue
                
                # Check if this is a master cell
                if (r, c) in master_cells:
                    info = master_cells[(r, c)]
                    rowspan = info['rowspan']
                    colspan = info['colspan']
                    
                    if rowspan > 1 and colspan > 1:
                        tokens.extend(['<td', f' rowspan="{rowspan}"', f' colspan="{colspan}"', '>', '</td>'])
                    elif rowspan > 1:
                        tokens.extend(['<td', f' rowspan="{rowspan}"', '>', '</td>'])
                    elif colspan > 1:
                        tokens.extend(['<td', f' colspan="{colspan}"', '>', '</td>'])
                else:
                    # Regular cell
                    tokens.extend(['<td>', '</td>'])
            
            tokens.append('</tr>')
        
        tokens.append('</table>')
        
        return {
            'structure': {
                'tokens': tokens
            }
        }
