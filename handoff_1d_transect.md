# Satvision-Pix4D 1D Transect Adaptation: Progress & Handoff

This document summarizes the work completed to adapt the `satvision-pix4d-base` (SatMAE ViT) model for 1D CloudSat/ABI transect data, and outlines the exact steps remaining for a future agent to complete the adaptation.

## Current State

The goal is to adapt the existing 2D SatMAE model (which expects `512x512` spatial tiles) to process 1D cloud transect/profile data shaped `512x1` (512 footprints along a ground track, 1 nearest pixel wide) with 16 channels and 7 timesteps, and predict cloud properties across 40 vertical altitude bins.

The work was divided into three phases based on the `1d_transect_adaptation_plan.md`.

*   **Phase 1: Architecture & Config Changes** ✅ (COMPLETED)
*   **Phase 2: Pretrained Weights Loading** ✅ (COMPLETED)
*   **Phase 3: Finetuning Pipeline (Dataset, Decoder, Training)** ✅ (COMPLETED)

## What Has Been Completed (Phase 3)

The entire downstream finetuning pipeline has been built and smoke-tested (using a mock encoder to simulate the not-yet-adapted SatMAE). The following files were created/modified in `/home/aliewehr/satvision-pix4d`:

1.  **`satvision_pix4d/datasets/transect_dataset.py` (NEW)**
    *   `TransectDataset`: PyTorch `Dataset` that parses the `512x1x16x7` `.npz` files produced by the new pipeline. Permutes the data to `(C, T, H, W)` = `(16, 7, 512, 1)` format required by the SatMAE encoder. Supports both binary (`cloud_binary_mask`) and multi-class (`cloud_class`) labels.
    *   `TransectDataModule`: PyTorch Lightning module for file discovery and train/val/test splitting.
    *   `satvision_pix4d/datasets/__init__.py` was updated to export these classes.

2.  **`satvision_pix4d/models/transect_model.py` (NEW)**
    *   `TransectDecoder`: A custom 1D UNet-style decoder (`Conv1d`, `ConvTranspose1d`). It takes the 32 spatial patch tokens (1024-dim) output by the encoder and upsamples them (32→64→128→256→512) to produce the final `(512, 40)` cloud curtain logits.
    *   `SatMAETransectModel`: Wraps the MAE encoder and the custom decoder. It handles dropping the CLS token and performing temporal mean-pooling across the 7 timesteps before passing the spatial tokens to the decoder.
    *   `TransectLightningModule`: PyTorch Lightning training module implementing Dice + CrossEntropy combined loss, differential learning rates (encoder vs decoder), and configurable encoder unfreezing schedules.

3.  **`scripts/finetune_transect.py` (NEW)**
    *   The main training script that wires together the pre-trained weights, the dataset module, and the Lightning model. Highly configurable via environment variables (data dirs, batch size, learning rates, checkpoint resuming, etc.).

4.  **`scripts/submit_finetune_transect.sh` (NEW)**
    *   A Slurm submission script template for running the finetuning job on the cluster (supports easy toggling between single-GPU and multi-GPU DDP).

**Smoke Tests:** The Phase 3 pipeline was successfully tested end-to-end using synthetic data and a mock encoder, verifying that shapes align perfectly: `Input (2, 16, 7, 512, 1)` → `Encoder Tokens (2, 32, 1024)` → `Decoder Output (2, 9, 512, 40)`.

---

## What Has Been Completed (Phases 1 & 2)

Before the training script could run with the real SatMAE model, the core MAE architecture needed to be tweaked to accept asymmetric 1D patches, and the pre-trained 2D weights needed to be reshaped to match.

### Phase 1: Architecture & Config Changes (COMPLETED)

1.  **Config Typing (`satvision_pix4d/configs/config.py`)**
    *   Changed `_C.DATA.IMG_SIZE = (224, 224)` and `_C.MODEL.MAE_VIT.PATCH_SIZE = (16, 16)`.
    *   Updated YAML configs to use lists for these parameters.
    *   Updated downstream code like `reconstruct_satmae.py` to extract the integer dimension to preserve backward compatibility.

2.  **Positional Embeddings (`satvision_pix4d/models/utils/pos_embed.py`)**
    *   Modified `get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)` to accept a tuple for `grid_size` and construct the meshgrid properly for asymmetric grids.

3.  **Patch Embedding (`satvision_pix4d/models/encoders/models_mae_temporal.py`)**
    *   Ensured `MaskedAutoencoderViT` supports asymmetric patches (e.g. `(16, 1)`).
    *   Updated `patchify`, `unpatchify`, `tokens_to_pixel_mask`, and the grid calculations in both encoder and decoder forward passes to unpack the height/width and dynamically calculate `grid_h, grid_w`.
    *   Updated `decoder_pred` linear layer to use `patch_h * patch_w`.

### Phase 2: Pretrained Weights Adaptation (COMPLETED)

1.  **Reshape the PatchEmbed Kernel (`scripts/finetune_transect.py`):**
    *   In `load_pretrained_mae`, intercepted the state dict and extracted the `patch_embed.proj.weight` tensor.
    *   Since the patch size changed to `(16, 1)`, the weights were averaged along the width dimension using `old_w.mean(dim=-1, keepdim=True)`.
    *   Mismatched keys not used for finetuning (like the pre-trained `decoder_pred`) were safely dropped.
    *   Programmatically overrode `config.DATA.IMG_SIZE = (512, 1)` and `config.MODEL.MAE_VIT.PATCH_SIZE = (16, 1)` in `main()` using `config.defrost()` prior to building the model.

**Next Steps:**
Everything is implemented and the finetuning script is ready to run on the cluster! You can now use `scripts/submit_finetune_transect.sh` to begin training.
