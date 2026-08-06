import sys
import torch
import torch.nn as nn
from satvision_pix4d.configs.config import _C
from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.models.transect_model import SatMAETransectModel, TransectDecoder

def main():
    print("Testing 1D Transect Model...")

    # Set up config for 1D
    config = _C.clone()
    config.defrost()
    config.DATA.IMG_SIZE = (512, 1)
    config.MODEL.MAE_VIT.PATCH_SIZE = (16, 1)
    # Using small values for fast testing
    config.MODEL.MAE_VIT.EMBED_DIM = 64
    config.MODEL.MAE_VIT.DEPTHS = 2
    config.MODEL.MAE_VIT.NUM_HEADS = 2
    config.MODEL.MAE_VIT.DECODER_EMBED_DIM = 32
    config.MODEL.MAE_VIT.DECODER_DEPTH = 1
    config.MODEL.MAE_VIT.DECODER_NUM_HEADS = 1
    config.MODEL.MAE_VIT.IN_CHANS = 16
    config.freeze()

    # Build encoder
    print("Building encoder...")
    mae = build_satmae_model(config)
    
    # Disable strict img size
    pe = mae.patch_embed
    if hasattr(pe, "strict_img_size"):
        pe.strict_img_size = False
    if hasattr(pe, "img_size"):
        pe.img_size = None
    if hasattr(pe, "dynamic_img_pad"):
        pe.dynamic_img_pad = True

    # Build transect wrapper
    print("Building transect model...")
    model = SatMAETransectModel(
        mae_model=mae,
        num_classes=9,
        num_bins=40,
        target_len=512,
        patch_size=config.MODEL.MAE_VIT.PATCH_SIZE,
        freeze_encoder=False,
        temporal_pool="mean"
    )

    # Test forward pass
    B, C, T, H, W = 2, 16, 7, 512, 1
    chips = torch.randn(B, C, T, H, W)
    print(f"Input shape: {chips.shape}")

    try:
        logits = model(chips)
        print(f"Output shape: {logits.shape}")
        assert logits.shape == (B, 9, 512, 40), f"Expected (2, 9, 512, 40), got {logits.shape}"
        print("Test passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
