import json
import os
import cv2
import numpy as np
from pathlib import Path


# ================= CONFIG =================
JSONL_PATH = r"D:\USTH\Computer_Vision\table\test.jsonl"
TABLE_IMG_DIR = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
ROW_OUT_DIR = r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\test"
COL_OUT_DIR = r"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\test"

os.makedirs(ROW_OUT_DIR, exist_ok=True)
os.makedirs(COL_OUT_DIR, exist_ok=True)


def parse_html_to_grid(tokens):
    grid = []
    row = -1
    col = 0
    active_rowspans = {}

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "<tr>":
            row += 1
            col = 0
            while col in active_rowspans and active_rowspans[col] >= row:
                col += 1

        elif tok == "<td>" or tok == "<td":
            rowspan = 1
            colspan = 1

            j = i + 1
            while j < len(tokens):
                t = tokens[j]

                if "rowspan" in t:
                    rowspan = int(t.split("\"")[1])
                if "colspan" in t:
                    colspan = int(t.split("\"")[1])

                # stop when next structural token starts
                if t in ("<td>", "<td", "<tr>", "</tr>", "</table>"):
                    break

                j += 1

            grid.append({
                "row_start": row,
                "row_end": row + rowspan - 1,
                "col_start": col,
                "col_end": col + colspan - 1,
                "rowspan": rowspan,
                "colspan": colspan
            })

            if rowspan > 1:
                for c in range(col, col + colspan):
                    active_rowspans[c] = row + rowspan - 1

            col += colspan
            while col in active_rowspans and active_rowspans[col] >= row:
                col += 1

        i += 1
        
    return grid


def adjust_bbox(bbox, table_bbox):
    tx1, ty1, _, _ = table_bbox
    x1, y1, x2, y2 = bbox
    return [
        int(x1 - tx1),
        int(y1 - ty1),
        int(x2 - tx1),
        int(y2 - ty1),
    ]


def generate_space_masks(img_shape, cells):
    H, W = img_shape[:2]
    row_mask = np.zeros((H, W), dtype=np.uint8)
    col_mask = np.zeros((H, W), dtype=np.uint8)

    max_row = max(c["row_end"] for c in cells)
    max_col = max(c["col_end"] for c in cells)

    # ---------- ROW SPACE ----------
    for r in range(max_row):
        upper = [
            c for c in cells
            if c["row_end"] == r and c["rowspan"] == 1
        ]
        lower = [
            c for c in cells
            if c["row_start"] == r + 1 and c["rowspan"] == 1
        ]

        if not upper or not lower:
            continue

        y_top = max(c["bbox"][3] for c in upper)
        y_bot = min(c["bbox"][1] for c in lower)

        if y_bot > y_top:
            row_mask[y_top:y_bot, :] = 255

    # ---------- COLUMN SPACE ----------
    for cidx in range(max_col):
        left = [
            c for c in cells
            if c["col_end"] == cidx and c["colspan"] == 1
        ]
        right = [
            c for c in cells
            if c["col_start"] == cidx + 1 and c["colspan"] == 1
        ]

        if not left or not right:
            continue

        x_l = max(c["bbox"][2] for c in left)
        x_r = min(c["bbox"][0] for c in right)

        if x_r > x_l:
            col_mask[:, x_l:x_r] = 255

    return row_mask, col_mask


def main():
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            table_id = obj["table_id"]

            img_path = os.path.join(TABLE_IMG_DIR, f"table_{table_id}.png")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            table_bbox = obj["bbox"]
            html_tokens = obj["html"]["structure"]["tokens"]
            html_cells = parse_html_to_grid(html_tokens)
            print("HTML cells:", len(html_cells))
            print("Text cells:", len(obj["html"]["cells"]))
            text_cells = obj["html"]["cells"]

            cells = []
            for grid_cell, text_cell in zip(html_cells, text_cells):
                if "bbox" not in text_cell:
                    continue

                adj_bbox = adjust_bbox(text_cell["bbox"], table_bbox)

                cells.append({
                    **grid_cell,
                    "bbox": adj_bbox
                })

            row_mask, col_mask = generate_space_masks(img.shape, cells)

            row_out_path = os.path.join(ROW_OUT_DIR, f"table_{table_id}.png")
            col_out_path = os.path.join(COL_OUT_DIR, f"table_{table_id}.png")

            cv2.imwrite(row_out_path, row_mask)
            cv2.imwrite(col_out_path, col_mask)


if __name__ == "__main__":
    main()
