import json
import os
from PIL import Image

# ================== CONFIG ==================
IMAGE_ROOT = r"D:\USTH\Computer_Vision\table\dataset\images\test"    
JSONL_PATH = r"D:\USTH\Computer_Vision\table\test.jsonl"
OUTPUT_DIR = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
# ============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue

        record = json.loads(line)

        table_id = record["table_id"]
        filename = record["filename"]      # ví dụ: train/MDT_2018_page_32.png
        bbox = record["bbox"]               # [x1, y1, x2, y2]

        rel = filename.replace("/", os.sep).lstrip(os.sep)
        img_path = os.path.join(IMAGE_ROOT, rel)

        if not os.path.exists(img_path):
            # try parent images folder (e.g. IMAGE_ROOT was ".../val" but filename has "train/...")
            alt = os.path.join(os.path.dirname(IMAGE_ROOT), rel)
            if os.path.exists(alt):
                img_path = alt
            else:
                # try using only the basename under IMAGE_ROOT
                base = os.path.join(IMAGE_ROOT, os.path.basename(rel))
                if os.path.exists(base):
                    img_path = base
                else:
                    print(f"[WARN] Image not found: {img_path}")
                    continue

        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Bbox table
        x1, y1, x2, y2 = bbox

        # Clamp để tránh lỗi vượt biên
        x1 = int(clamp(x1, 0, W - 1))
        y1 = int(clamp(y1, 0, H - 1))
        x2 = int(clamp(x2, 0, W))
        y2 = int(clamp(y2, 0, H))

        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Invalid bbox for table_id={table_id}")
            continue

        # Crop table
        table_img = img.crop((x1, y1, x2, y2))

        # Đặt tên theo table_id
        out_name = f"table_{table_id}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        table_img.save(out_path)

        if line_idx % 500 == 0:
            print(f"Processed {line_idx} tables...")

print("Done cropping all tables!")
