import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from table_dataset import TableDataset   
import csv
# ================= CONFIG =================
MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\row_space_seg.pth"
IMG_SIZE = 512
BATCH_SIZE = 8
THRESHOLD = 0.5
csv_path = r"D:\USTH\Computer_Vision\table\result_row_space_seg.csv"
IMG_DIR = r"D:\USTH\Computer_Vision\table\crop_table\images\test"
MASK_DIR = r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\test"

device = "cuda" if torch.cuda.is_available() else "cpu"

def resize_with_padding(img, mask, size=512, pad_val_img=255, pad_val_mask=0):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    img_canvas = np.full((size, size, 3), pad_val_img, dtype=img.dtype)
    mask_canvas = np.full((size, size), pad_val_mask, dtype=mask.dtype)

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2

    img_canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
    mask_canvas[y0:y0+new_h, x0:x0+new_w] = mask_resized

    return img_canvas, mask_canvas

# ================= METRICS =================
def segmentation_metrics(pred, target, eps=1e-7):
    """
    pred, target: (B,1,H,W) hoặc (B,H,W), binary {0,1}
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    TP = (pred * target).sum(dim=1)
    TN = ((1 - pred) * (1 - target)).sum(dim=1)
    FP = (pred * (1 - target)).sum(dim=1)
    FN = ((1 - pred) * target).sum(dim=1)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return {
        "accuracy": accuracy.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "dice": dice.mean(),
        "iou": iou.mean(),
    }

# ================= MODEL =================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# ================= DATA =================
test_dataset = TableDataset(
    img_dir=IMG_DIR,
    mask_dir=MASK_DIR,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ================= EVALUATE =================
metrics_sum = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "dice": 0.0,
    "iou": 0.0,
}
num_batches = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > THRESHOLD).float()

        batch_metrics = segmentation_metrics(preds, masks)

        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k].item()

        num_batches += 1

# ================= PRINT RESULT =================
print("\n" + "=" * 50)
print("TEST METRICS")

for k in metrics_sum:
    metrics_sum[k] /= max(num_batches, 1)
    print(f"{k.capitalize():10s}: {metrics_sum[k]:.4f}")
    
# ================= SAVE RESULT TO CSV =================


# chuẩn hóa lại metrics (đã chia trung bình ở bước print)
metrics_avg = {}
for k in metrics_sum:
    metrics_avg[k] = metrics_sum[k]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])  # header

    for k, v in metrics_avg.items():
        writer.writerow([k, f"{v:.6f}"])

print(f"\nSaved metrics to: {csv_path}")