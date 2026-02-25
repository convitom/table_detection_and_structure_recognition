"""
Space Segmentation Module
Input: Cropped table image
Output: Row space mask and Column space mask
"""
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from itertools import groupby


class SpaceSegmentor:
    def __init__(self, row_model_path, col_model_path, device=None, img_size=512):
        """
        Initialize space segmentor
        
        Args:
            row_model_path: Path to row space segmentation model
            col_model_path: Path to col space segmentation model
            device: 'cuda' or 'cpu'
            img_size: Input image size for model
        """
        self.img_size = img_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build and load models
        self.row_model = self._build_model()
        self.col_model = self._build_model()
        
        self.row_model = self._auto_load_model(self.row_model, row_model_path)
        self.col_model = self._auto_load_model(self.col_model, col_model_path)
    
    def _build_model(self):
        """Build U-Net model"""
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    
    def _auto_load_model(self, model, path):
        """Load model weights"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        else:
            model = ckpt
        
        model.to(self.device)
        model.eval()
        return model
    
    def _resize_with_padding(self, img, pad_val=255):
        """Resize image with padding to square"""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.full((self.img_size, self.img_size, 3), pad_val, dtype=img.dtype)
        
        x0 = (self.img_size - new_w) // 2
        y0 = (self.img_size - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
        
        meta = {
            "orig_h": h,
            "orig_w": w,
            "x0": x0,
            "y0": y0,
            "new_h": new_h,
            "new_w": new_w,
        }
        return canvas, meta
    
    def _recover_mask(self, mask512, meta):
        """Recover mask to original size"""
        x0, y0 = meta["x0"], meta["y0"]
        new_h, new_w = meta["new_h"], meta["new_w"]
        
        mask_crop = mask512[y0:y0+new_h, x0:x0+new_w]
        
        mask_final = cv2.resize(
            mask_crop,
            (meta["orig_w"], meta["orig_h"]),
            interpolation=cv2.INTER_NEAREST
        )
        return mask_final
    
    def _straighten_row_mask(self, mask, min_ratio=0.75):
        """Clean and straighten row mask"""
        H, W = mask.shape
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        row_sum = mask.sum(axis=1)
        valid_rows = np.where(row_sum >= min_ratio * W)[0]
        
        clean = np.zeros_like(mask)
        
        for k, g in groupby(enumerate(valid_rows), lambda x: x[0] - x[1]):
            rows = [x[1] for x in g]
            clean[min(rows):max(rows)+1, :] = 1
        
        return clean
    
    def _straighten_col_mask(self, mask, min_ratio=0.75):
        """Clean and straighten column mask"""
        H, W = mask.shape
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        col_sum = mask.sum(axis=0)
        valid_cols = np.where(col_sum >= min_ratio * H)[0]
        
        clean = np.zeros_like(mask)
        
        for k, g in groupby(enumerate(valid_cols), lambda x: x[0] - x[1]):
            cols = [x[1] for x in g]
            clean[:, min(cols):max(cols)+1] = 1
        
        return clean
    
    def segment(self, img, threshold=0.2):
        """
        Segment row and column spaces
        
        Args:
            img: Input image (RGB or BGR)
            threshold: Threshold for binary prediction (default: 0.2)
            
        Returns:
            tuple: (row_mask, col_mask) - both are binary masks (0 or 1)
            Note: This returns raw predictions. Apply _straighten_row_mask() and 
            _straighten_col_mask() separately if you need custom min_ratio cleaning.
        """
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Preprocess
        img_pad, meta = self._resize_with_padding(img)
        img_tensor = img_pad.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            # Row prediction
            row_prob = torch.sigmoid(self.row_model(img_tensor))
            row_pred = (row_prob > threshold).float()
            
            # Column prediction
            col_prob = torch.sigmoid(self.col_model(img_tensor))
            col_pred = (col_prob > threshold).float()
        
        # Recover to original size
        row_mask_raw = self._recover_mask(row_pred.squeeze().cpu().numpy(), meta)
        col_mask_raw = self._recover_mask(col_pred.squeeze().cpu().numpy(), meta)
        
        # Return raw masks (user can apply cleaning separately with custom min_ratio)
        return row_mask_raw.astype(np.uint8), col_mask_raw.astype(np.uint8)
