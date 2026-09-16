import logging
import torch.nn as nn
from functools import partial
from satvision_pix4d.models.encoders.models_mae_temporal import \
    MaskedAutoencoderViT as MaskedAutoencoderViTTemporal
from satvision_pix4d.models.utils.precision_support import FP32LayerNorm

# Encoder presets; the lightweight decoder remains configurable.
MODEL_SIZES = {
    "330m": (1024, 24, 16),
    "700m": (1280, 34, 20),
    "3b": (2560, 38, 32),
    "25b": (6144, 55, 48),
}

# -----------------------------------------------------------------------------
# build_satmae_model
# -----------------------------------------------------------------------------
def build_satmae_model(config):
    """Builds the satmae model.

    Args:
        config: config object

    Raises:
        NotImplementedError: if the model is
        not swinv2, then this will be thrown.

    Returns:
        MiMModel: masked-image-modeling model
    """
    model_type = config.MODEL.TYPE
    if model_type == 'satmae':
        cfg = config.MODEL.MAE_VIT
        size = cfg.SIZE.lower()
        if size == "custom":
            width, depth, heads = cfg.EMBED_DIM, cfg.DEPTHS, cfg.NUM_HEADS
        elif size in MODEL_SIZES:
            width, depth, heads = MODEL_SIZES[size]
        else:
            raise ValueError(f"Unknown MAE size {size!r}; choose custom, 330m, 700m, 3b, or 25b")
        components = config.DATA.TEMPORAL_COMPONENTS
        if (not components or len(set(components)) != len(components)
                or any(c not in {"year", "month", "day", "hour", "minute"} for c in components)):
            raise ValueError("TEMPORAL_COMPONENTS must be unique supported timestamp field names")
        lower, upper = [], []
        if cfg.LOSS_TYPE == "pixel_structure":
            stats = [config.DATA.MEAN, config.DATA.STD, config.DATA.MIN, config.DATA.MAX]
            if any(len(v) != cfg.IN_CHANS for v in stats):
                raise ValueError("MEAN, STD, MIN and MAX must match MAE_VIT.IN_CHANS")
            if any(s <= 0 for s in config.DATA.STD):
                raise ValueError("Channel standard deviations must be positive")
            lower = [(v-m)/s for v,m,s in zip(config.DATA.MIN, config.DATA.MEAN, config.DATA.STD)]
            upper = [(v-m)/s for v,m,s in zip(config.DATA.MAX, config.DATA.MEAN, config.DATA.STD)]
        model = MaskedAutoencoderViTTemporal(
            img_size=config.DATA.IMG_SIZE,
            n_time_components=len(components),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            loss_type=cfg.LOSS_TYPE,
            visnir_channels=cfg.VISNIR_CHANNELS,
            ssim_min=lower,
            ssim_max=upper,
            patch_size=config.MODEL.MAE_VIT.PATCH_SIZE,
            in_chans=config.MODEL.MAE_VIT.IN_CHANS,
            embed_dim=width,
            depth=depth,
            num_heads=heads,
            decoder_embed_dim=config.MODEL.MAE_VIT.DECODER_EMBED_DIM,
            decoder_depth=config.MODEL.MAE_VIT.DECODER_DEPTH,
            decoder_num_heads=config.MODEL.MAE_VIT.DECODER_NUM_HEADS,
            mlp_ratio=config.MODEL.MAE_VIT.MLP_RATIO,
            norm_layer=partial(FP32LayerNorm, eps=1e-6),
            norm_pix_loss=config.MODEL.MAE_VIT.NORM_PIX_LOSS,
            same_mask=config.MODEL.MAE_VIT.SAME_MASK,
            visible_loss_weight=config.MODEL.MAE_VIT.VISIBLE_LOSS_WEIGHT,
            refine_pixels=config.MODEL.MAE_VIT.REFINE_PIXELS,
            refinement_channels=config.MODEL.MAE_VIT.REFINEMENT_CHANNELS,
            refinement_depth=config.MODEL.MAE_VIT.REFINEMENT_DEPTH,
        )

        logging.info("MAE size=%s, parameters=%s", size, f"{sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters()):,}")
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
