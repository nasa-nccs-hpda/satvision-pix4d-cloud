import logging
import torch.nn as nn
from functools import partial
from satvision_pix4d.models.encoders.models_mae_temporal import \
    MaskedAutoencoderViT as MaskedAutoencoderViTTemporal
from satvision_pix4d.models.utils.precision_support import FP32LayerNorm

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
        model = MaskedAutoencoderViTTemporal(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.MAE_VIT.PATCH_SIZE,
            in_chans=config.MODEL.MAE_VIT.IN_CHANS,
            embed_dim=config.MODEL.MAE_VIT.EMBED_DIM,
            depth=config.MODEL.MAE_VIT.DEPTHS,
            num_heads=config.MODEL.MAE_VIT.NUM_HEADS,
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

        logging.info(str(model))
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
