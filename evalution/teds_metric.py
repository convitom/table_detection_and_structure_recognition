"""
TEDS (Tree Edit Distance based Similarity) Metric
Measures similarity between predicted and ground truth table structures

TEDS-Simple: Structure-only comparison (grid without span information)
TEDS-Complex: Full structure comparison (grid + rowspan/colspan attributes)
"""
from apted import APTED
from apted.helpers import Tree
from lxml import etree, html
from collections import deque


class TEDSMetric:
    def __init__(self, structure_only=False, ignore_nodes=None):
        """
        Initialize TEDS metric
        
        Args:
            structure_only: If True, compute TEDS-Struct (ignore cell content)
            ignore_nodes: List of node types to ignore (default: None)
        """
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes if ignore_nodes else []
    
    def _tokens_to_html_string(self, tokens):
        """Convert token list to HTML string"""
        return ''.join(tokens)
    
    def _normalize_html(self, html_string):
        """Normalize HTML string"""
        # Remove whitespace
        html_string = html_string.strip()
        
        # Parse and serialize to normalize
        try:
            tree = html.fromstring(html_string)
            normalized = html.tostring(tree, encoding='unicode')
            return normalized
        except:
            # If parsing fails, return original
            return html_string
    
    def _html_to_tree(self, html_string):
        """
        Convert HTML string to tree structure for APTED
        
        Returns:
            Tree object for APTED
        """
        try:
            root = html.fromstring(html_string)
        except:
            # If parsing fails, create empty tree
            return Tree('table')
        
        def build_tree(node):
            """Recursively build tree from HTML node"""
            # Get tag name
            tag = node.tag
            
            # Skip ignored nodes
            if tag in self.ignore_nodes:
                return None
            
            # For structure-only mode, ignore text content
            if self.structure_only:
                label = tag
            else:
                # Include text content in label
                text = (node.text or '').strip()
                label = f"{tag}:{text}" if text else tag
            
            # Build children
            children = []
            for child in node:
                child_tree = build_tree(child)
                if child_tree is not None:
                    children.append(child_tree)
            
            # Create tree node
            if children:
                return Tree(label, *children)
            else:
                return Tree(label)
        
        return build_tree(root)
    
    def _normalize_tree(self, tree_str):
        """Normalize tree string representation"""
        # Remove extra whitespace
        return ' '.join(tree_str.split())
    
    def _expand_to_grid(self, html_dict):
        """
        Expand merged cells to show full grid structure for TEDS-Simple
        
        Example:
            Input:  <tr><td colspan="2"></td><td></td></tr>
            Output: <tr><td></td><td></td><td></td></tr>
        
        This requires:
        1. Parse rowspan/colspan attributes
        2. Track which cells are "occupied" by spans
        3. Rebuild with all grid cells explicit
        
        Args:
            html_dict: HTML structure dict with tokens
            
        Returns:
            HTML dict with all grid cells as individual <td> tags
        """
        if isinstance(html_dict, dict) and 'structure' in html_dict and 'tokens' in html_dict['structure']:
            tokens = html_dict['structure']['tokens']
        else:
            return html_dict
        
        # Parse HTML to understand structure
        import re
        
        # First pass: collect rows and their cells with span info
        rows = []
        current_row = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '<tr>':
                current_row = []
                i += 1
            
            elif token == '</tr>':
                if current_row:
                    rows.append(current_row)
                i += 1
            
            elif token == '<td' or token == '<td>':
                # Parse cell with potential rowspan/colspan
                rowspan = 1
                colspan = 1
                
                if token == '<td':
                    i += 1
                    # Parse attributes
                    while i < len(tokens) and tokens[i] != '>':
                        attr = tokens[i]
                        
                        # Extract rowspan
                        match = re.search(r'rowspan="?(\d+)"?', attr)
                        if match:
                            rowspan = int(match.group(1))
                        
                        # Extract colspan
                        match = re.search(r'colspan="?(\d+)"?', attr)
                        if match:
                            colspan = int(match.group(1))
                        
                        i += 1
                    i += 1  # Skip '>'
                else:
                    i += 1  # Skip '<td>'
                
                # Skip content until </td>
                while i < len(tokens) and tokens[i] != '</td>':
                    i += 1
                i += 1  # Skip '</td>'
                
                # Add cell info
                current_row.append({
                    'rowspan': rowspan,
                    'colspan': colspan
                })
            
            else:
                i += 1
        
        # Second pass: build grid considering rowspan
        # This is complex because rowspan affects future rows
        
        if not rows:
            return html_dict
        
        # Build occupation map: which cells are occupied by rowspan from above
        max_cols = 0
        for row in rows:
            cols = sum(cell['colspan'] for cell in row)
            max_cols = max(max_cols, cols)
        
        # Track occupation: occupied[r][c] = True if occupied
        occupied = [[False] * max_cols for _ in range(len(rows))]
        
        # Process each row and mark occupations
        for r_idx, row in enumerate(rows):
            col_pos = 0
            
            for cell in row:
                # Skip occupied columns
                while col_pos < max_cols and occupied[r_idx][col_pos]:
                    col_pos += 1
                
                # Mark this cell's occupation
                for dr in range(cell['rowspan']):
                    for dc in range(cell['colspan']):
                        if r_idx + dr < len(rows) and col_pos + dc < max_cols:
                            occupied[r_idx + dr][col_pos + dc] = True
                
                col_pos += cell['colspan']
        
        # Third pass: generate simple grid HTML
        simple_tokens = ['<table>']
        
        for r_idx in range(len(rows)):
            simple_tokens.append('<tr>')
            
            # Count actual columns in this row
            num_cols = sum(1 for c in range(max_cols) if occupied[r_idx][c])
            
            for _ in range(num_cols):
                simple_tokens.extend(['<td>', '</td>'])
            
            simple_tokens.append('</tr>')
        
        simple_tokens.append('</table>')
        
        return {
            'structure': {
                'tokens': simple_tokens
            }
        }
    
    def compute_teds(self, pred_html, gt_html, simple_mode=False):
        """
        Compute TEDS score between predicted and ground truth HTML
        
        Args:
            pred_html: Predicted HTML (dict with 'structure'->'tokens' or string)
            gt_html: Ground truth HTML (dict with 'structure'->'tokens' or string)
            simple_mode: If True, rebuild as simple grid (TEDS-Simple)
        
        Returns:
            float: TEDS score (0-1, higher is better)
        """
        # Expand to full grid if in simple mode
        if simple_mode:
            pred_html = self._expand_to_grid(pred_html)
            gt_html = self._expand_to_grid(gt_html)
        
        # Convert to HTML strings
        if isinstance(pred_html, dict):
            if 'structure' in pred_html and 'tokens' in pred_html['structure']:
                pred_str = self._tokens_to_html_string(pred_html['structure']['tokens'])
            else:
                pred_str = str(pred_html)
        else:
            pred_str = str(pred_html)
        
        if isinstance(gt_html, dict):
            if 'structure' in gt_html and 'tokens' in gt_html['structure']:
                gt_str = self._tokens_to_html_string(gt_html['structure']['tokens'])
            else:
                gt_str = str(gt_html)
        else:
            gt_str = str(gt_html)
        
        # Normalize HTML
        pred_str = self._normalize_html(pred_str)
        gt_str = self._normalize_html(gt_str)
        
        # Convert to trees
        pred_tree = self._html_to_tree(pred_str)
        gt_tree = self._html_to_tree(gt_str)
        
        # Compute tree edit distance
        apted = APTED(pred_tree, gt_tree)
        ted = apted.compute_edit_distance()
        
        # Compute TEDS score
        # TEDS = 1 - (normalized_TED)
        # normalized_TED = TED / max(|T_pred|, |T_gt|)
        # where |T| is the number of nodes in tree T
        
        n_nodes_pred = self._count_nodes(pred_tree)
        n_nodes_gt = self._count_nodes(gt_tree)
        
        max_nodes = max(n_nodes_pred, n_nodes_gt)
        
        if max_nodes == 0:
            return 1.0  # Both trees are empty
        
        normalized_ted = ted / max_nodes
        teds = max(0.0, 1.0 - normalized_ted)
        
        return teds
    
    def _count_nodes(self, tree):
        """Count number of nodes in tree"""
        if tree is None:
            return 0
        
        count = 1  # Count root
        
        # Count children
        if hasattr(tree, 'children'):
            for child in tree.children:
                count += self._count_nodes(child)
        
        return count
    
    def compute_batch_teds(self, pred_htmls, gt_htmls, simple_mode=False):
        """
        Compute TEDS for a batch of predictions
        
        Args:
            pred_htmls: List of predicted HTML structures
            gt_htmls: List of ground truth HTML structures
            simple_mode: If True, use TEDS-Simple (no span attributes)
        
        Returns:
            dict: {
                'scores': list of individual TEDS scores,
                'mean': mean TEDS score,
                'median': median TEDS score
            }
        """
        if len(pred_htmls) != len(gt_htmls):
            raise ValueError("Number of predictions and ground truths must match")
        
        scores = []
        for pred, gt in zip(pred_htmls, gt_htmls):
            score = self.compute_teds(pred, gt, simple_mode=simple_mode)
            scores.append(score)
        
        import numpy as np
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
