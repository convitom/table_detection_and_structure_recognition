# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH_SIZE = 1

MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\span_seg_2.pth"
OUT_DIR = r"D:\USTH\Computer_Vision\table\eva\span_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Eval mode --------
DO_PR_CURVE = False
FIXED_THRESHOLD = 0.1

THRESHOLDS = (
    np.linspace(0.05, 0.95, 19)
    if DO_PR_CURVE
    else [FIXED_THRESHOLD]
)

# -------- Overlay config --------
SAVE_OVERLAY = True
OVERLAY_ALPHA = 0.4
OVERLAY_MAX_IMAGES = 20000

OVERLAY_DIR = os.path.join(OUT_DIR, "overlay")
ROW_DIR = os.path.join(OVERLAY_DIR, "row_span")
COL_DIR = os.path.join(OVERLAY_DIR, "col_span")

if SAVE_OVERLAY:
    os.makedirs(ROW_DIR, exist_ok=True)
    os.makedirs(COL_DIR, exist_ok=True)

# =========================
# UTILS
# =========================
def resize_pad(img, size=512, pad_val=255, is_mask=False):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = (
        np.full((size, size, 3), pad_val, dtype=img.dtype)
        if img.ndim == 3
        else np.full((size, size), pad_val, dtype=img.dtype)
    )

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized
    return canvas


def draw_gt_contour(img, mask, color):
    """
    mask: (H, W) binary
    """
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        img,
        cnts,
        -1,
        color,
        thickness=1  # mỏng
    )


def save_overlay(img_tensor, prob, gt, threshold, fname):
    """
    img_tensor: (3,H,W) float
    prob: (2,H,W)
    gt: (2,H,W)
    """
    img = (img_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    overlay = img.copy()

    # ---------- Prediction ----------
    row_pred = prob[0] >= threshold
    col_pred = prob[1] >= threshold

    overlay[row_pred] = (
        (1 - OVERLAY_ALPHA) * overlay[row_pred]
        + OVERLAY_ALPHA * np.array([255, 0, 0])
    )
    overlay[col_pred] = (
        (1 - OVERLAY_ALPHA) * overlay[col_pred]
        + OVERLAY_ALPHA * np.array([0, 255, 0])
    )

    overlay = overlay.astype(np.uint8)

    # ---------- GT contour ----------
    if gt[0].sum() > 0:
        draw_gt_contour(
            overlay,
            gt[0],
            color=(180, 0, 0)  # đỏ đậm
        )

    if gt[1].sum() > 0:
        draw_gt_contour(
            overlay,
            gt[1],
            color=(0, 180, 0)  # xanh đậm
        )

    out_name = f"{os.path.splitext(fname)[0]}_t{threshold:.2f}.png"
    out_img = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # ---------- Save theo GT ----------
    if gt[0].sum() > 0:
        cv2.imwrite(os.path.join(ROW_DIR, out_name), out_img)

    if gt[1].sum() > 0:
        cv2.imwrite(os.path.join(COL_DIR, out_name), out_img)


# =========================
# DATASET
# =========================
class SpanTableDataset(Dataset):
    def __init__(self, img_dir, row_space_dir, col_space_dir, row_span_dir, col_span_dir):
        self.img_dir = img_dir
        self.row_space_dir = row_space_dir
        self.col_space_dir = col_space_dir
        self.row_span_dir = row_span_dir
        self.col_span_dir = col_span_dir

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            fname = self.files[idx]

            img = cv2.imread(os.path.join(self.img_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            row_space = cv2.imread(os.path.join(self.row_space_dir, fname), 0)
            col_space = cv2.imread(os.path.join(self.col_space_dir, fname), 0)
            row_span = cv2.imread(os.path.join(self.row_span_dir, fname), 0)
            col_span = cv2.imread(os.path.join(self.col_span_dir, fname), 0)

            img = resize_pad(img, IMG_SIZE, 255, False)
            row_space = resize_pad(row_space, IMG_SIZE, 0, True)
            col_space = resize_pad(col_space, IMG_SIZE, 0, True)
            row_span = resize_pad(row_span, IMG_SIZE, 0, True)
            col_span = resize_pad(col_span, IMG_SIZE, 0, True)

            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)

            x = np.concatenate([
                img,
                (row_space > 0)[None],
                (col_space > 0)[None]
            ], axis=0)

            y = np.stack([
                (row_span > 0),
                (col_span > 0)
            ], axis=0)

            return torch.tensor(x), torch.tensor(y), fname

        except:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys, fnames = zip(*batch)
    return torch.stack(xs), torch.stack(ys), fnames


# =========================
# LOAD DATA & MODEL
# =========================
test_dataset = SpanTableDataset(
    img_dir=r"D:\USTH\Computer_Vision\table\crop_table\images\test",
    row_space_dir=r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\test",
    col_space_dir=r"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\test",
    row_span_dir=r"D:\USTH\Computer_Vision\table\crop_table\row_span_masks\test",
    col_span_dir=r"D:\USTH\Computer_Vision\table\crop_table\col_span_masks\test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_skip_none
)

def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=2
    ).to(DEVICE)

model = build_model()
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.eval()

# =========================
# EVALUATION
# =========================
saved_overlay = 0

for t in tqdm(THRESHOLDS, desc="Eval"):
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            imgs, masks, fnames = batch
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            masks = masks.numpy()

            for b in range(imgs.size(0)):
                if (
                    SAVE_OVERLAY
                    and saved_overlay < OVERLAY_MAX_IMAGES
                    and (masks[b, 0].sum() > 0 or masks[b, 1].sum() > 0)
                ):
                    save_overlay(
                        imgs[b].cpu().numpy()[:3],
                        probs[b],
                        masks[b],
                        t,
                        fnames[b]
                    )
                    saved_overlay += 1

print("✅ DONE – overlay row/col + GT contour chuẩn chỉnh")
