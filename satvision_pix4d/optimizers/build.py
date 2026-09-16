import torch
import logging

from functools import partial


OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
}


# -----------------------------------------------------------------------------
# get_optimizer_from_dict
# -----------------------------------------------------------------------------
def get_optimizer_from_dict(optimizer_name, config):
    """Gets the proper optimizer given an optimizer name.

    Args:
        optimizer_name (str): name of the optimizer
        config: config object

    Raises:
        KeyError: thrown if loss key is not present in dict

    Returns:
        loss: pytorch optimizer
    """

    if optimizer_name.lower() == "lamb":
        from satvision_pix4d.optimizers.lamb import Lamb
        return Lamb
    if optimizer_name.lower() in {"fusedlamb", "fusedadamw"}:
        import deepspeed
        return (deepspeed.ops.lamb.FusedLamb if optimizer_name.lower() == "fusedlamb"
                else deepspeed.ops.adam.FusedAdam)

    try:

        optimizer_to_use = OPTIMIZERS[optimizer_name.lower()]

    except KeyError:

        error_msg = f"{optimizer_name} is not an implemented optimizer"

        error_msg = f"{error_msg}. Available optimizer functions: {OPTIMIZERS.keys()}"  # noqa: E501

        raise KeyError(error_msg)

    return optimizer_to_use


# -----------------------------------------------------------------------------
# build_optimizer
# -----------------------------------------------------------------------------
def build_optimizer(config, model, is_pretrain=False):
    """
    Build optimizer for Swin/SwinV2 and SatMAE.
    """
    logging.info('>>>>>>>>>> Build Optimizer')

    skip = {}
    skip_keywords = {}
    optimizer_name = config.TRAIN.OPTIMIZER.NAME

    logging.info(f'Building optimizer: {optimizer_name}')

    optimizer_to_use = get_optimizer_from_dict(optimizer_name, config)

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    # -----------------------------
    # Swin / SwinV2
    # -----------------------------
    if config.MODEL.TYPE.lower() in ["swin", "swinv2"]:
        logging.info("Detected Swin/SwinV2 configuration.")

        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            if config.MODEL.TYPE == 'swin':
                depths = config.MODEL.SWIN.DEPTHS
            else:
                depths = config.MODEL.SWINV2.DEPTHS

            num_layers = sum(depths)

            get_layer_func = partial(
                get_swin_layer,
                num_layers=num_layers + 2,
                depths=depths
            )

            scales = [
                config.TRAIN.LAYER_DECAY ** i
                for i in reversed(range(num_layers + 2))
            ]

            parameters = get_finetune_param_groups(
                model,
                config.TRAIN.BASE_LR,
                config.TRAIN.WEIGHT_DECAY,
                get_layer_func,
                scales,
                skip,
                skip_keywords
            )

    # -----------------------------
    # SatMAE (or ViT)
    # -----------------------------
    elif config.MODEL.TYPE.lower() in ["satmae", "mae_vit", "vit"]:
        logging.info("Detected SatMAE / ViT configuration.")

        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            # For SatMAE, no per-layer decay or get_layer_func
            parameters = get_finetune_param_groups(
                model,
                config.TRAIN.BASE_LR,
                config.TRAIN.WEIGHT_DECAY,
                get_layer_func=None,
                scales=None,
                skip=skip,
                skip_keywords=skip_keywords
            )
    else:
        raise ValueError(
            f"Unsupported MODEL.TYPE '{config.MODEL.TYPE}'. "
            "Please check your config."
        )

    # Build the optimizer
    optimizer = optimizer_to_use(
        parameters,
        eps=config.TRAIN.OPTIMIZER.EPS,
        betas=config.TRAIN.OPTIMIZER.BETAS,
        lr=config.TRAIN.BASE_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )

    logging.info(optimizer)
    return optimizer


# -----------------------------------------------------------------------------
# get_finetune_param_groups
# -----------------------------------------------------------------------------
def get_finetune_param_groups(model,
                              lr,
                              weight_decay,
                              get_layer_func,
                              scales,
                              skip_list=(),
                              skip_keywords=()):

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias") \
            or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.

        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)

        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }

            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


# -----------------------------------------------------------------------------
# check_keywords_in_name
# -----------------------------------------------------------------------------
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# -----------------------------------------------------------------------------
# get_pretrain_param_groups
# -----------------------------------------------------------------------------
def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):

    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue

        if len(param.shape) == 1 or name.endswith(".bias") or \
            (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            no_decay.append(param)

            no_decay_name.append(name)

        else:

            has_decay.append(param)

            has_decay_name.append(name)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


# -----------------------------------------------------------------------------
# get_swin_layer
# -----------------------------------------------------------------------------
def get_swin_layer(name, num_layers, depths):

    if name in ("mask_token"):

        return 0

    elif name.startswith("patch_embed"):

        return 0

    elif name.startswith("layers"):

        layer_id = int(name.split('.')[1])

        block_id = name.split('.')[3]

        if block_id == 'reduction' or block_id == 'norm':

            return sum(depths[:layer_id + 1])

        layer_id = sum(depths[:layer_id]) + int(block_id)

        return layer_id + 1

    else:

        return num_layers - 1
