import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

suckmydik = "test"
table_id = 337
MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\span_seg_2.pth"
img_path = r"D:\USTH\Computer_Vision\table\crop_table\images\{suckmydik}\table_{table_id}.png".format(suckmydik=suckmydik,table_id=table_id)
row_space_path = r"D:\USTH\Computer_Vision\table\crop_table\row_space_masks\{suckmydik}\table_{table_id}.png".format(suckmydik=suckmydik,table_id=table_id)
col_space_path = r"D:\USTH\Computer_Vision\table\crop_table\col_space_masks\{suckmydik}\table_{table_id}.png".format(suckmydik=suckmydik,table_id=table_id)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
            print("✔ Loaded model_state")
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            print("✔ Loaded state_dict")
        else:
            model.load_state_dict(ckpt)
            print("✔ Loaded raw state_dict")
    else:
        model = ckpt
        print("✔ Loaded full model")

    model.to(device)
    model.eval()
    return model

model = auto_load_model(build_model(), MODEL_PATH, DEVICE)

IMG_SIZE = 512

def resize_with_padding_single(img, size=512, pad_val=255):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)
    if img.ndim == 3:
        canvas = np.full((size, size, 3), pad_val, dtype=img.dtype)
    else:
        canvas = np.full((size, size), pad_val, dtype=img.dtype)

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = img_resized

    return canvas, (x0, y0, new_w, new_h)

def resize_with_padding_with_meta(img, meta, size=512, pad_val=0):
    x0, y0, new_w, new_h = meta

    if img.ndim == 3:
        canvas = np.full((size, size, 3), pad_val, dtype=img.dtype)
    else:
        canvas = np.full((size, size), pad_val, dtype=img.dtype)

    img_resized = cv2.resize(img, (new_w, new_h),
                             cv2.INTER_AREA if img.ndim == 3 else cv2.INTER_NEAREST)

    canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
    return canvas

def infer_span(
    img_path,
    row_space_path,
    col_space_path,
    thresh=0.5
):
    # ===== LOAD IMAGE =====
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"❌ Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    row_space = cv2.imread(row_space_path, cv2.IMREAD_GRAYSCALE)
    if row_space is None:
        raise FileNotFoundError(f"❌ Cannot read row_space: {row_space_path}")
    col_space = cv2.imread(col_space_path, cv2.IMREAD_GRAYSCALE)
    if col_space is None:
        raise FileNotFoundError(f"❌ Cannot read col_space: {col_space_path}")
    H0, W0 = img_rgb.shape[:2]

    # ===== RESIZE + PAD =====
    img_pad, meta = resize_with_padding_single(img_rgb, IMG_SIZE, 255)
    row_pad = resize_with_padding_with_meta(row_space, meta, IMG_SIZE, 0)
    col_pad = resize_with_padding_with_meta(col_space, meta, IMG_SIZE, 0)

    # ===== NORMALIZE =====
    img_pad = img_pad.astype(np.float32) / 255.0
    row_pad = (row_pad > 0).astype(np.float32)
    col_pad = (col_pad > 0).astype(np.float32)

    img_pad = img_pad.transpose(2, 0, 1)  # (3,H,W)

    x = np.concatenate([
        img_pad,
        row_pad[None],
        col_pad[None]
    ], axis=0)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ===== INFERENCE =====
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    print("logits min/max:",
        logits.min().item(),
        logits.max().item())

    print("probs min/max:",
        probs.min().item(),
        probs.max().item())
    row_span = (probs[0] > thresh).astype(np.uint8)
    col_span = (probs[1] > thresh).astype(np.uint8)

    # ===== UNPAD → ORIGINAL SIZE =====
    x0, y0, w, h = meta
    row_span = row_span[y0:y0+h, x0:x0+w]
    col_span = col_span[y0:y0+h, x0:x0+w]

    row_span = cv2.resize(row_span, (W0, H0), cv2.INTER_NEAREST)
    col_span = cv2.resize(col_span, (W0, H0), cv2.INTER_NEAREST)
    print("meta:", meta)
    print("row_pad sum:", row_pad.sum())
    print("col_pad sum:", col_pad.sum())
    return img_rgb, row_span, col_span

def overlay_span(img, row_span, col_span, alpha=0.45):
    overlay = img.copy()

    red = np.zeros_like(img)
    red[..., 0] = 255

    blue = np.zeros_like(img)
    blue[..., 2] = 255

    overlay[row_span == 1] = (
        overlay[row_span == 1] * (1 - alpha) + red[row_span == 1] * alpha
    ).astype(np.uint8)

    overlay[col_span == 1] = (
        overlay[col_span == 1] * (1 - alpha) + blue[col_span == 1] * alpha
    ).astype(np.uint8)

    return overlay

img, row_span, col_span = infer_span(
    img_path=img_path,
    row_space_path=row_space_path,
    col_space_path=col_space_path
)

overlay = overlay_span(img, row_span, col_span)

plt.figure(figsize=(14,6))
plt.subplot(1,3,1)
plt.title("row-span")
plt.imshow(row_span, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("col-span")
plt.imshow(col_span, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay, cmap="gray")
plt.axis("off")

plt.show()

