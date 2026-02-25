import json
import fitz
from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\USTH\Computer_Vision\table")
TRAIN_DIR = ROOT / "val"
JSONL_PATH = ROOT / "val.jsonl"

DPI = 72
SCALE = DPI / 72

def pdf_to_png_and_get_img_height(pdf_path, png_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(png_path)

    return pix.height   # <<< PIXEL HEIGHT (QUAN TRỌNG)

def convert_bbox_pdf_to_img(bbox, img_height):
    x0, y0, x1, y1 = bbox
    return [
        x0 * SCALE,
        img_height - y1 * SCALE,
        x1 * SCALE,
        img_height - y0 * SCALE
    ]

page_height_cache = {}
new_lines = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        # ---- filename gốc (pdf) ----
        pdf_rel = data["filename"]           # train/.../page_xx.pdf
        pdf_path = ROOT / pdf_rel
        pdf_name = Path(pdf_rel).stem
        png_rel = f"train/{pdf_name}.png"
        png_path = TRAIN_DIR / f"{pdf_name}.png"

        # ---- render pdf → png (1 lần) ----
        if pdf_name not in page_height_cache:
            img_h = pdf_to_png_and_get_img_height(pdf_path, png_path)
            page_height_cache[pdf_name] = img_h
            pdf_path.unlink()
        else:
            img_h = page_height_cache[pdf_name]

        # ---- update filename ----
        data["filename"] = png_rel

        # ---- TABLE bbox ----
        if "bbox" in data:
            data["bbox"] = convert_bbox_pdf_to_img(data["bbox"], img_h)

        # ---- CELL bbox ----
        for cell in data.get("html", {}).get("cells", []):
            if "bbox" in cell:
                cell["bbox"] = convert_bbox_pdf_to_img(cell["bbox"], img_h)

        new_lines.append(json.dumps(data, ensure_ascii=False))

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines) + "\n")

print("DONE:")
print(f"- Rendered {len(page_height_cache)} PDFs → PNG")
print("- Deleted all PDFs")
print("- BBox scaled + Y-axis flipped correctly")
print("- train.jsonl overwritten (PNG only)")
