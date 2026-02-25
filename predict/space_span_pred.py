import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ================= CONFIG =================
suckmydik = "test"
table_id = 337

MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\span_seg_2.pth"
img_path = rf"D:\USTH\Computer_Vision\table\crop_table\images\{suckmydik}\table_{table_id}.png"
row_space_path = rf"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\{suckmydik}\table_{table_id}.png"
col_space_path = rf"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\{suckmydik}\table_{table_id}.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

# ================= MODEL =================
def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=2
    ).to(DEVICE)

def auto_load_model(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model = ckpt

    model.to(device)
    model.eval()
    return model

model = auto_load_model(build_model(), MODEL_PATH, DEVICE)

# ================= RESIZE + PAD =================
def resize_with_padding_single(img, size=512, pad_val=255):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)

    canvas = np.full(
        (size, size, img.shape[2]) if img.ndim == 3 else (size, size),
        pad_val,
        dtype=img.dtype
    )

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized

    return canvas, (x0, y0, new_w, new_h)

def resize_with_padding_with_meta(img, meta, size=512, pad_val=0):
    x0, y0, new_w, new_h = meta

    canvas = np.full(
        (size, size, img.shape[2]) if img.ndim == 3 else (size, size),
        pad_val,
        dtype=img.dtype
    )

    interp = cv2.INTER_AREA if img.ndim == 3 else cv2.INTER_NEAREST
    img_resized = cv2.resize(img, (new_w, new_h), interp)

    canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized
    return canvas

# ================= INFERENCE =================
def infer_span(img_path, row_space_path, col_space_path, thresh=0.5):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    row_space = cv2.imread(row_space_path, cv2.IMREAD_GRAYSCALE)
    col_space = cv2.imread(col_space_path, cv2.IMREAD_GRAYSCALE)
    if row_space is None or col_space is None:
        raise FileNotFoundError("space mask missing")

    H0, W0 = img_rgb.shape[:2]

    img_pad, meta = resize_with_padding_single(img_rgb, IMG_SIZE, 255)
    row_pad = resize_with_padding_with_meta(row_space, meta, IMG_SIZE, 0)
    col_pad = resize_with_padding_with_meta(col_space, meta, IMG_SIZE, 0)

    img_pad = img_pad.astype(np.float32) / 255.0
    row_pad = (row_pad > 0).astype(np.float32)
    col_pad = (col_pad > 0).astype(np.float32)

    img_pad = img_pad.transpose(2, 0, 1)

    x = np.concatenate([
        img_pad,
        row_pad[None],
        col_pad[None]
    ], axis=0)

    x = torch.tensor(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()

    row_span = (probs[0] > thresh).astype(np.uint8)
    col_span = (probs[1] > thresh).astype(np.uint8)

    x0, y0, w, h = meta

    row_span = row_span[y0:y0+h, x0:x0+w]
    col_span = col_span[y0:y0+h, x0:x0+w]
    row_space = row_pad[y0:y0+h, x0:x0+w]
    col_space = col_pad[y0:y0+h, x0:x0+w]

    row_span = cv2.resize(row_span, (W0, H0), cv2.INTER_NEAREST)
    col_span = cv2.resize(col_span, (W0, H0), cv2.INTER_NEAREST)
    row_space = cv2.resize(row_space, (W0, H0), cv2.INTER_NEAREST)
    col_space = cv2.resize(col_space, (W0, H0), cv2.INTER_NEAREST)

    return img_rgb, row_span, col_span, row_space, col_space

# ================= OVERLAY =================
def overlay_all(img, row_span, col_span, row_space, col_space,
                alpha_span=0.5, alpha_space=0.35):

    overlay = img.copy()

    RED    = np.array([255,   0,   0], np.uint8)   # row-span
    BLUE   = np.array([  0,   0, 255], np.uint8)   # col-span
    GREEN  = np.array([  0, 255,   0], np.uint8)   # row-space
    YELLOW = np.array([255, 255,   0], np.uint8)   # col-space

    overlay[row_space > 0] = (
        overlay[row_space > 0] * (1 - alpha_space) + GREEN * alpha_space
    ).astype(np.uint8)

    overlay[col_space > 0] = (
        overlay[col_space > 0] * (1 - alpha_space) + YELLOW * alpha_space
    ).astype(np.uint8)

    overlay[row_span == 1] = (
        overlay[row_span == 1] * (1 - alpha_span) + RED * alpha_span
    ).astype(np.uint8)

    overlay[col_span == 1] = (
        overlay[col_span == 1] * (1 - alpha_span) + BLUE * alpha_span
    ).astype(np.uint8)

    return overlay

# ================= RUN =================
img, row_span, col_span, row_space, col_space = infer_span(
    img_path,
    row_space_path,
    col_space_path
)

overlay = overlay_all(img, row_span, col_span, row_space, col_space)

# ================= VIS =================
plt.figure(figsize=(18,8))

plt.subplot(2,3,1)
plt.title("Row space")
plt.imshow(row_space, cmap="gray")
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Col space")
plt.imshow(col_space, cmap="gray")
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Row span")
plt.imshow(row_span, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Col span")
plt.imshow(col_span, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay (ALL)")
plt.imshow(overlay)
plt.axis("off")

plt.show()
