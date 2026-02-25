# -*- coding: utf-8 -*-
import sys
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
table_id = 337
MODEL_PATH = r"D:\USTH\Computer_Vision\table\model\table_row_space_seg.pth"
IMAGE_PATH = r"D:\USTH\Computer_Vision\table\crop_table\images\test\table_{table_id}.png".format(table_id=table_id)

IMG_SIZE = 512
THRESHOLD = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================
def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

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

model = auto_load_model(build_model(), MODEL_PATH, device)

# ================= PREPROCESS =================
def resize_with_padding_infer(img, size=512, pad_val=255):
    h, w = img.shape[:2]
    scale = size / max(h, w)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    img_resized = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    canvas = np.full((size, size, 3), pad_val, dtype=img.dtype)

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2

    canvas[y0:y0+new_h, x0:x0+new_w] = img_resized

    meta = {
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
        "x0": x0,
        "y0": y0,
        "new_h": new_h,
        "new_w": new_w
    }
    return canvas, meta

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot read image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_pad, meta = resize_with_padding_infer(img, IMG_SIZE)

    img_tensor = img_pad.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)

    return img, img_tensor, meta

# ================= POSTPROCESS =================
def recover_mask(mask512, meta):
    """
    mask512: (512,512) numpy float {0,1}
    """
    x0, y0 = meta["x0"], meta["y0"]
    new_w, new_h = meta["new_w"], meta["new_h"]

    # remove padding
    mask_crop = mask512[y0:y0+new_h, x0:x0+new_w]

    # resize back to original image
    mask_final = cv2.resize(
        mask_crop,
        (meta["orig_w"], meta["orig_h"]),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_final

def overlay_mask(image, mask, alpha=0.4):
    overlay = image.copy()

    red = np.zeros_like(image)
    red[..., 0] = 255

    mask = mask.astype(bool)
    overlay[mask] = cv2.addWeighted(
        image[mask], 1 - alpha, red[mask], alpha, 0
    )
    return overlay

# ================= INFERENCE =================
orig_img, img_tensor, meta = preprocess_image(IMAGE_PATH)
img_tensor = img_tensor.to(device)

with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    pred = (probs > THRESHOLD).float()

mask512 = pred.squeeze().cpu().numpy()
mask = recover_mask(mask512, meta)

overlay = overlay_mask(orig_img, mask)

# ================= VISUALIZE =================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(orig_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
