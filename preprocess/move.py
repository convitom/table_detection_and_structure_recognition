import json
import shutil
from pathlib import Path

# ====== CONFIG ======
ROOT_DIR = Path(r"D:\USTH\Computer_Vision\table")
OUT_DIR = Path(r"D:\USTH\Computer_Vision\table")
JSONL_FILES = {
    "train": "FinTabNet_1.0.0_cell_train.jsonl",
    "val":   "FinTabNet_1.0.0_cell_val.jsonl",
    "test":  "FinTabNet_1.0.0_cell_test.jsonl",
}
# ====================

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

def make_new_pdf_name(old_rel_path: str) -> str:
    """
    AVB/2005/page_53.pdf -> AVB_2005_page_53.pdf
    """
    return old_rel_path.replace("/", "_")

def process_split(split, jsonl_path):
    out_jsonl = OUT_DIR / f"{split}.jsonl"
    moved_pdfs = set()

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(out_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)
            old_rel_path = data["filename"]
            old_pdf_path = ROOT_DIR / old_rel_path

            if not old_pdf_path.exists():
                continue

            new_pdf_name = make_new_pdf_name(old_rel_path)
            new_rel_path = f"{split}/{new_pdf_name}"
            new_pdf_path = OUT_DIR / new_rel_path

            # Move mỗi PDF đúng 1 lần
            if old_rel_path not in moved_pdfs:
                shutil.move(str(old_pdf_path), str(new_pdf_path))
                moved_pdfs.add(old_rel_path)

            data["filename"] = new_rel_path
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"[{split}] moved {len(moved_pdfs)} pdfs → {out_jsonl}")

def main():
    ensure_dirs()
    for split, jsonl_file in JSONL_FILES.items():
        print(f"Processing {split}...")
        process_split(split, jsonl_file)

if __name__ == "__main__":
    main()
