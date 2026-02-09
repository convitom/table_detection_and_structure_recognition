import torch
import segmentation_models_pytorch as smp
import os

# ===== CONFIG =====
CKPT_PATH = r"D:\USTH\Computer_Vision\table\model\span_seg_2.pth"          # checkpoint gốc
OUT_PATH  = r"D:\USTH\Computer_Vision\table\model\span_seg.pth"  # file output
DEVICE = "cpu"                        # convert thì cpu cho nhẹ
# ==================


def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=5,
        classes=2
    )


def extract_state_dict(ckpt):
    """
    Trả về state_dict chuẩn từ mọi loại checkpoint
    """
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            print("✔ Found model_state")
            return ckpt["model_state"]
        elif "state_dict" in ckpt:
            print("✔ Found state_dict")
            return ckpt["state_dict"]
        else:
            # dict nhưng bản thân nó đã là state_dict
            print("✔ Treat checkpoint as raw state_dict")
            return ckpt
    else:
        raise ValueError("❌ Checkpoint là full model object, không convert trực tiếp được")


def main():
    print(f"📦 Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    state_dict = extract_state_dict(ckpt)

    # optional: verify load được
    print("🔍 Verifying state_dict with model architecture...")
    model = build_model()
    model.load_state_dict(state_dict, strict=True)

    # save weight-only
    torch.save(state_dict, OUT_PATH)
    print(f"✅ Saved weight-only model to: {OUT_PATH}")

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"📉 Weight-only size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
