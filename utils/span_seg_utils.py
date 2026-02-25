"""
Span Segmentation Module
Input: Cropped table image + row space mask + col space mask
Output: Row span mask and Column span mask
"""
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp


class SpanSegmentor:
    def __init__(self, model_path, device=None, img_size=512):
        """
        Initialize span segmentor
        
        Args:
            model_path: Path to span segmentation model
            device: 'cuda' or 'cpu'
            img_size: Input image size for model
        """
        self.img_size = img_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build and load model
        self.model = self._build_model()
        self.model = self._auto_load_model(self.model, model_path)
    
    def _build_model(self):
        """Build U-Net model with 5 input channels"""
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=5,  # 3 for image + 1 for row_space + 1 for col_space
            classes=2  # row_span and col_span
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
    
    def _resize_with_padding_single(self, img, pad_val=255):
        """Resize image with padding to square"""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        
        img_resized = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)
        if img.ndim == 3:
            canvas = np.full((self.img_size, self.img_size, 3), pad_val, dtype=img.dtype)
        else:
            canvas = np.full((self.img_size, self.img_size), pad_val, dtype=img.dtype)
        
        x0 = (self.img_size - new_w) // 2
        y0 = (self.img_size - new_h) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
        
        return canvas, (x0, y0, new_w, new_h)
    
    def _resize_with_padding_with_meta(self, img, meta, pad_val=0):
        """Resize image with same padding as original"""
        x0, y0, new_w, new_h = meta
        
        if img.ndim == 3:
            canvas = np.full((self.img_size, self.img_size, 3), pad_val, dtype=img.dtype)
        else:
            canvas = np.full((self.img_size, self.img_size), pad_val, dtype=img.dtype)
        
        img_resized = cv2.resize(img, (new_w, new_h),
                                 cv2.INTER_AREA if img.ndim == 3 else cv2.INTER_NEAREST)
        
        canvas[y0:y0+new_h, x0:x0+new_w] = img_resized
        return canvas
    
    def segment(self, img, row_space_mask, col_space_mask, threshold=0.5):
        """
        Segment row and column spans
        
        Args:
            img: Input table image (RGB or BGR)
            row_space_mask: Row space mask from SpaceSegmentor
            col_space_mask: Column space mask from SpaceSegmentor
            threshold: Threshold for binary prediction
            
        Returns:
            tuple: (row_span_mask, col_span_mask) - both are binary masks (0 or 1)
        """
        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        H0, W0 = img.shape[:2]
        
        # Resize with padding
        img_pad, meta = self._resize_with_padding_single(img, 255)
        row_pad = self._resize_with_padding_with_meta(row_space_mask, meta, 0)
        col_pad = self._resize_with_padding_with_meta(col_space_mask, meta, 0)
        
        # Normalize
        img_pad = img_pad.astype(np.float32) / 255.0
        row_pad = (row_pad > 0).astype(np.float32)
        col_pad = (col_pad > 0).astype(np.float32)
        
        img_pad = img_pad.transpose(2, 0, 1)  # (3,H,W)
        
        # Concatenate: 3 image channels + 1 row_space + 1 col_space
        x = np.concatenate([
            img_pad,
            row_pad[None],
            col_pad[None]
        ], axis=0)
        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        row_span = (probs[0] > threshold).astype(np.uint8)
        col_span = (probs[1] > threshold).astype(np.uint8)
        
        # Unpad and resize to original size
        x0, y0, w, h = meta
        row_span = row_span[y0:y0+h, x0:x0+w]
        col_span = col_span[y0:y0+h, x0:x0+w]
        
        row_span = cv2.resize(row_span, (W0, H0), cv2.INTER_NEAREST)
        col_span = cv2.resize(col_span, (W0, H0), cv2.INTER_NEAREST)
        
        return row_span, col_span
