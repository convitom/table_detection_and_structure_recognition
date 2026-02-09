"""
Table Reconstruction Module
Input: Table image + 4 masks (row_space, col_space, row_span, col_span)
Output: Reconstructed table structure with grid lines
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict


class TableReconstructor:
    def __init__(self, use_ocr=True, ocr_lang="en"):
        """
        Initialize table reconstructor
        
        Args:
            use_ocr: Whether to use OCR for text detection
            ocr_lang: Language for OCR
        """
        self.use_ocr = use_ocr
        if use_ocr:
            self.ocr = PaddleOCR(det=True, rec=False, cls=False, lang=ocr_lang, show_log=False)
        else:
            self.ocr = None
    
    def _extract_lines_from_mask(self, mask, orientation='horizontal', min_length=20):
        """Extract separator lines from mask"""
        binary = (mask > 127).astype(np.uint8) * 255
        
        if orientation == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            projection = np.sum(lines_mask, axis=1)
            positions = np.where(projection > min_length * 255)[0]
        else:  # vertical
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            projection = np.sum(lines_mask, axis=0)
            positions = np.where(projection > min_length * 255)[0]
        
        if len(positions) == 0:
            return []
        
        groups = np.split(positions, np.where(np.diff(positions) > 5)[0] + 1)
        lines = [int(np.mean(g)) for g in groups if len(g) > 0]
        
        return sorted(lines)
    
    def _build_grid(self, row_space_mask, col_space_mask, img_shape, min_length=20):
        """Build grid from space masks"""
        h, w = img_shape[:2]
        
        row_lines = self._extract_lines_from_mask(row_space_mask, 'horizontal', min_length)
        col_lines = self._extract_lines_from_mask(col_space_mask, 'vertical', min_length)
        
        # Add boundaries
        if not row_lines or row_lines[0] > 5:
            row_lines.insert(0, 0)
        if not row_lines or row_lines[-1] < h - 5:
            row_lines.append(h)
        if not col_lines or col_lines[0] > 5:
            col_lines.insert(0, 0)
        if not col_lines or col_lines[-1] < w - 5:
            col_lines.append(w)
        
        return row_lines, col_lines
    
    def _identify_span_regions(self, span_mask, row_lines, col_lines, threshold=0.85):
        """Identify which cells are span cells"""
        binary_span = (span_mask > 127).astype(np.uint8)
        refined_span = np.zeros_like(binary_span)
        
        for i in range(len(row_lines) - 1):
            for j in range(len(col_lines) - 1):
                y1, y2 = row_lines[i], row_lines[i+1]
                x1, x2 = col_lines[j], col_lines[j+1]
                
                cell = binary_span[y1:y2, x1:x2]
                cell_area = (y2 - y1) * (x2 - x1)
                span_area = np.sum(cell)
                
                if span_area > threshold * cell_area:
                    refined_span[y1:y2, x1:x2] = 1
        
        return refined_span
    
    def _find_span_blobs(self, refined_span):
        """Find connected components of span regions"""
        num_labels, labels = cv2.connectedComponents(refined_span.astype(np.uint8))
        
        blobs = []
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[0])
                blobs.append({
                    'label': label,
                    'mask': mask,
                    'bbox': (x, y, x+w, y+h)
                })
        
        return blobs
    
    def _detect_text_in_blob(self, img, bbox, blob_type, row_lines):
        """Detect text regions in a span blob using OCR"""
        if not self.use_ocr or self.ocr is None:
            return []
        
        bx1, by1, bx2, by2 = bbox
        crop = img[by1:by2, bx1:bx2]
        
        try:
            result = self.ocr.ocr(crop, cls=False, rec=False)
            if result is None or result[0] is None:
                return []
            
            text_bboxes = []
            for line in result[0]:
                pts = np.array(line).reshape(-1, 2)
                pts[:, 0] += bx1
                pts[:, 1] += by1
                text_bboxes.append(pts.tolist())
            
            return text_bboxes
        except:
            return []
    
    def _find_best_separators_combination(self, text_bboxes, candidates, blob_range, axis='col'):
        """Find best separator positions that don't cut through text"""
        if len(text_bboxes) == 0 or len(candidates) == 0:
            return []
        
        # Convert text boxes to rectangles
        text_rects = []
        for box in text_bboxes:
            pts = np.array(box)
            if axis == 'col':
                x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
                text_rects.append((x_min, x_max))
            else:  # row
                y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
                text_rects.append((y_min, y_max))
        
        # Find candidates that don't intersect text
        valid_separators = []
        for sep in candidates:
            is_valid = True
            for t_min, t_max in text_rects:
                if t_min < sep < t_max:
                    is_valid = False
                    break
            if is_valid:
                valid_separators.append(sep)
        
        return valid_separators
    
    def reconstruct(self, img, row_space_mask, col_space_mask, 
                   row_span_mask, col_span_mask, min_length=20, span_threshold=0.85):
        """
        Reconstruct table structure
        
        Args:
            img: Table image (BGR or RGB)
            row_space_mask: Row space mask
            col_space_mask: Column space mask
            row_span_mask: Row span mask
            col_span_mask: Column span mask
            min_length: Min length for line extraction (default: 20)
            span_threshold: Threshold for span region identification (default: 0.85)
            
        Returns:
            dict with keys:
                - 'grid': (row_lines, col_lines)
                - 'span_blobs': list of span blob info
        """
        # Build grid
        row_lines, col_lines = self._build_grid(row_space_mask, col_space_mask, img.shape, min_length)
        
        all_span_blobs = []
        
        # Process column span cells
        col_span_refined = self._identify_span_regions(col_span_mask, row_lines, col_lines, span_threshold)
        col_span_blobs = self._find_span_blobs(col_span_refined)
        
        for blob in col_span_blobs:
            text_bboxes = self._detect_text_in_blob(img, blob['bbox'], 'col_span', row_lines)
            
            bx1, by1, bx2, by2 = blob['bbox']
            internal_cols = [c for c in col_lines if bx1 < c < bx2]
            
            best_seps = self._find_best_separators_combination(
                text_bboxes, internal_cols, (bx1, bx2), axis='col'
            )
            
            all_span_blobs.append({
                'type': 'col_span',
                'bbox': blob['bbox'],
                'best_separators': set(best_seps)
            })
        
        # Process row span cells
        row_span_refined = self._identify_span_regions(row_span_mask, row_lines, col_lines, span_threshold)
        row_span_blobs = self._find_span_blobs(row_span_refined)
        
        for blob in row_span_blobs:
            text_bboxes = self._detect_text_in_blob(img, blob['bbox'], 'row_span', row_lines)
            
            bx1, by1, bx2, by2 = blob['bbox']
            internal_rows = [r for r in row_lines if by1 < r < by2]
            
            best_seps = self._find_best_separators_combination(
                text_bboxes, internal_rows, (by1, by2), axis='row'
            )
            
            all_span_blobs.append({
                'type': 'row_span',
                'bbox': blob['bbox'],
                'best_separators': set(best_seps)
            })
        
        return {
            'grid': (row_lines, col_lines),
            'span_blobs': all_span_blobs
        }
    
    def _get_skip_segments(self, line_pos, span_blobs, line_type):
        """Find segments of line to skip (inside span blobs but not best separators)"""
        skip_segments = []
        
        for blob in span_blobs:
            bx1, by1, bx2, by2 = blob['bbox']
            best_seps = blob['best_separators']
            
            if line_type == 'row' and blob['type'] == 'row_span':
                if by1 < line_pos < by2:
                    if line_pos not in best_seps:
                        skip_segments.append((bx1, bx2))
            
            elif line_type == 'col' and blob['type'] == 'col_span':
                if bx1 < line_pos < bx2:
                    if line_pos not in best_seps:
                        skip_segments.append((by1, by2))
        
        return skip_segments
    
    def _draw_line_with_gaps(self, vis, line_pos, total_length, skip_segments, 
                            line_type, color=(0, 0, 255), thickness=1):
        """Draw line with gaps"""
        # Merge overlapping segments
        if skip_segments:
            skip_segments = sorted(skip_segments)
            merged = [skip_segments[0]]
            for start, end in skip_segments[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            skip_segments = merged
        
        # Create draw segments
        draw_segments = []
        current_pos = 0
        
        for skip_start, skip_end in skip_segments:
            if current_pos < skip_start:
                draw_segments.append((current_pos, skip_start))
            current_pos = skip_end
        
        if current_pos < total_length:
            draw_segments.append((current_pos, total_length))
        
        # Draw each segment
        if line_type == 'row':
            for start_x, end_x in draw_segments:
                cv2.line(vis, (int(start_x), line_pos), (int(end_x), line_pos), color, thickness)
        else:
            for start_y, end_y in draw_segments:
                cv2.line(vis, (line_pos, int(start_y)), (line_pos, int(end_y)), color, thickness)
    
    def draw_grid(self, img, result, color=(0, 0, 255), thickness=1):
        """
        Draw grid lines on image
        
        Args:
            img: Original image
            result: Result dict from reconstruct()
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with grid lines drawn
        """
        vis = img.copy()
        h, w = vis.shape[:2]
        
        row_lines, col_lines = result['grid']
        span_blobs = result['span_blobs']
        
        # Draw row lines (horizontal)
        for y in row_lines:
            skip_segments = self._get_skip_segments(y, span_blobs, 'row')
            
            if not skip_segments:
                cv2.line(vis, (0, y), (w, y), color, thickness)
            else:
                self._draw_line_with_gaps(vis, y, w, skip_segments, 'row', color, thickness)
        
        # Draw column lines (vertical)
        for x in col_lines:
            skip_segments = self._get_skip_segments(x, span_blobs, 'col')
            
            if not skip_segments:
                cv2.line(vis, (x, 0), (x, h), color, thickness)
            else:
                self._draw_line_with_gaps(vis, x, h, skip_segments, 'col', color, thickness)
        
        return vis
