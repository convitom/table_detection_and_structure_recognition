import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

IMG_SIZE = 512   # table nên >=256, 512 là ổn

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


class TableDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.mask_dir, fname),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None or mask is None:
            return None

        if img.size == 0 or mask.size == 0:
            return None
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return None
        # -------- SCALE AUGMENT --------
        if self.augment:
            scale = np.random.uniform(1, 1.2)
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_NEAREST)

        # -------- RESIZE + PAD --------
        img, mask = resize_with_padding(img, mask, IMG_SIZE)

        # -------- NORMALIZE --------
        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        # -------- TO TENSOR --------
        img = img.transpose(2, 0, 1)
        mask = mask[None, :, :]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )
