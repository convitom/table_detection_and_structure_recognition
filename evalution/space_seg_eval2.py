import os
import csv
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import groupby
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)   # .../table
sys.path.insert(0, PROJECT_ROOT)
from table_dataset import TableDataset   # <-- sửa đúng path


# ================= CONFIG =================
IMG_SIZE = 512
BATCH_SIZE = 8
THRESHOLDS = np.linspace(0.0, 1.0, 51)

RESULT_DIR = r"D:\USTH\Computer_Vision\table\evalution\space_seg_result"
os.makedirs(RESULT_DIR, exist_ok=True)

ROW_MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\row_space_seg.pth"
COL_MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\col_space_seg.pth"

ROW_IMG_DIR = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
ROW_MASK_DIR = r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\test"

COL_IMG_DIR = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
COL_MASK_DIR = r"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\test"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= POST PROCESS =================
def straighten_row_mask(mask, min_ratio=0.75):
    H, W = mask.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    row_sum = mask.sum(axis=1)
    valid_rows = np.where(row_sum >= min_ratio * W)[0]

    clean = np.zeros_like(mask)
    for k, g in groupby(enumerate(valid_rows), lambda x: x[0] - x[1]):
        rows = [x[1] for x in g]
        clean[min(rows):max(rows) + 1, :] = 1

    return clean


def straighten_col_mask(mask, min_ratio=0.75):
    H, W = mask.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    col_sum = mask.sum(axis=0)
    valid_cols = np.where(col_sum >= min_ratio * H)[0]

    clean = np.zeros_like(mask)
    for k, g in groupby(enumerate(valid_cols), lambda x: x[0] - x[1]):
        cols = [x[1] for x in g]
        clean[:, min(cols):max(cols) + 1] = 1

    return clean

# ================= METRICS =================
def compute_metrics(pred, target, mode, eps=1e-7):
    """
    pred, target: (B,1,H,W) binary
    mode: 'row' | 'col'
    """
    B = pred.size(0)

    pred = pred.squeeze(1).cpu().numpy()
    target = target.squeeze(1).cpu().numpy()

    pred_clean = np.zeros_like(pred)

    for i in range(B):
        if mode == "row":
            pred_clean[i] = straighten_row_mask(pred[i])
        else:
            pred_clean[i] = straighten_col_mask(pred[i])

    pred = torch.from_numpy(pred_clean).float()
    target = torch.from_numpy(target).float()

    pred = pred.view(B, -1)
    target = target.view(B, -1)

    TP = (pred * target).sum(dim=1)
    FP = (pred * (1 - target)).sum(dim=1)
    FN = ((1 - pred) * target).sum(dim=1)

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)

    return (
        precision.mean().item(),
        recall.mean().item(),
        dice.mean().item()
    )

# ================= MODEL =================
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# ================= EVALUATION =================
def evaluate_space(model, dataloader, name, mode):
    results = []

    with torch.no_grad():
        for th in tqdm(THRESHOLDS, desc=f"Evaluating {name}"):
            p_sum, r_sum, d_sum, n = 0, 0, 0, 0

            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)

                probs = torch.sigmoid(model(images))
                preds = (probs > th).float()

                p, r, d = compute_metrics(preds, masks, mode)

                p_sum += p
                r_sum += r
                d_sum += d
                n += 1

            results.append({
                "threshold": float(th),
                "precision": p_sum / n,
                "recall": r_sum / n,
                "dice": d_sum / n
            })

    # ---- Save CSV ----
    csv_path = os.path.join(RESULT_DIR, f"{name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ---- Best threshold ----
    best = max(results, key=lambda x: x["dice"])

    # ---- PR Curve ----
    recalls = [x["recall"] for x in results]
    precisions = [x["precision"] for x in results]

    plt.figure()
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve – {name}")
    plt.grid(True)

    fig_path = os.path.join(RESULT_DIR, f"{name}_pr_curve.png")
    plt.savefig(fig_path)
    plt.close()

    return best

# ================= DATA =================
row_loader = DataLoader(
    TableDataset(ROW_IMG_DIR, ROW_MASK_DIR),
    batch_size=BATCH_SIZE,
    shuffle=False
)

col_loader = DataLoader(
    TableDataset(COL_IMG_DIR, COL_MASK_DIR),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= RUN =================
row_model = load_model(ROW_MODEL_PATH)
col_model = load_model(COL_MODEL_PATH)

best_row = evaluate_space(
    row_model,
    row_loader,
    name="row_space",
    mode="row"
)

best_col = evaluate_space(
    col_model,
    col_loader,
    name="col_space",
    mode="col"
)

# ================= SAVE BEST =================
with open(os.path.join(RESULT_DIR, "best_thresholds.txt"), "w") as f:
    f.write("BEST THRESHOLDS (by Dice / F1)\n")
    f.write("=" * 40 + "\n")
    f.write(
        f"Row space: threshold={best_row['threshold']:.2f}, "
        f"Dice={best_row['dice']:.4f}\n"
    )
    f.write(
        f"Col space: threshold={best_col['threshold']:.2f}, "
        f"Dice={best_col['dice']:.4f}\n"
    )

print("\nDONE!")
print("Best Row:", best_row)
print("Best Col:", best_col)
