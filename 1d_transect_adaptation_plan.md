# Plan: Adapt satvision-pix4d-base (SatMAE ViT) for 1D CloudSat/ABI Transects

## Background & Resolved Questions

The preprocessing pipeline now produces **1D transect/profile** data shaped `512×1×16×7` (512 CloudSat footprints × 1 nearest ABI pixel × 16 channels × 7 timesteps), instead of the previous 2D spatial tiles (`512×512×16×7`).

Based on recent clarifications:
1. **Architecture:** The `satvision-pix4d-base` model is a **Masked Autoencoder built on a ViT** (SatMAE). We will reuse the existing `MaskedAutoencoderViTTemporal` code in the repository rather than using `timm` or writing a custom model from scratch.
2. **HuggingFace Checkpoint:** The weights are stored in `mp_rank_00_model_states.pt` on a gated HuggingFace repository.
3. **Temporal Dimension:** The 7 timesteps will not be compressed into channels. The MAE encoder will process the spatio-temporal tokens, and its output features can later be reshaped and fed into a downstream architecture like a **3D U-Net** for time-series processing.

---

## Goal 1: Modify the Base Architecture & Config for 1D `512×1` Input

### Step 1.1 — Update Configuration Typing in `config.py`

**File:** [`satvision_pix4d/configs/config.py`](file:///home/aliewehr/satvision-pix4d/satvision_pix4d/configs/config.py)

**Problem:** `yacs` enforces strict type matching. The current defaults are scalar integers:
```python
_C.DATA.IMG_SIZE = 224            # int
_C.MODEL.MAE_VIT.PATCH_SIZE = 16 # int
```
Overriding `IMG_SIZE` with `[512, 1]` from a YAML file triggers a `TypeError`.

**Change:** Convert these defaults to tuples/lists so YAML overrides like `[512, 1]` are accepted:
```python
_C.DATA.IMG_SIZE = (224, 224)
_C.MODEL.MAE_VIT.PATCH_SIZE = (16, 16)
```

### Step 1.2 — Adapt Patch Embedding for Asymmetric (1D) Input

**File:** [`satvision_pix4d/models/encoders/models_mae_temporal.py`](file:///home/aliewehr/satvision-pix4d/satvision_pix4d/models/encoders/models_mae_temporal.py)

The standard `PatchEmbed` expects square patches. For a `512×1` input, this must become asymmetric.
We will modify the `PatchEmbed` implementation to support tuple patch sizes (e.g., `(16, 1)`):
```python
# The patch embedding kernel and stride will become:
nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_h, 1), stride=(tubelet_size, patch_h, 1))
```

### Step 1.3 — Adapt Positional Embeddings for 1D Spatial + Temporal

**File:** [`satvision_pix4d/models/utils/pos_embed.py`](file:///home/aliewehr/satvision-pix4d/satvision_pix4d/models/utils/pos_embed.py)

The existing `get_2d_sincos_pos_embed` creates a 2D grid. For a `512×1` input, this degenerates to a 1D sequence in space.
We will add a conditional path:
- If the spatial width is `1` (i.e., 1D transect), use `get_1d_sincos_pos_embed_from_grid_torch` for the spatial dimension.
- Combine this with the existing 1D temporal positional embedding to create the final 3D (Time + Space 1D) positional embedding.

### Step 1.4 — Downstream Feature Extraction (3D U-Net Prep)

Since the plan is to eventually use a 3D U-Net, we will ensure that the forward pass of the ViT encoder can return the unflattened features.
The output of the ViT encoder (sequence of tokens) will be reshaped from `(B, T * (H/patch_h) * (W/patch_w), C)` back to a spatio-temporal grid `(B, C, T, H_out, W_out)` so it can be seamlessly passed to a 3D U-Net.

---

## Goal 2: Load Pretrained Weights from HuggingFace

### Step 2.1 — Download Weights via `huggingface_hub`

Since the repository is gated, we will require HuggingFace authentication. We will write a loading script that uses the user's token:

```python
from huggingface_hub import hf_hub_download

# Prompts for token or uses the CLI login
model_path = hf_hub_download(
    repo_id="nasa-cisto-data-science-group/satvision-pix4d-base",
    filename="mp_rank_00_model_states.pt",
    use_auth_token=True # Assumes user has run `huggingface-cli login` or provides token
)
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
```

### Step 2.2 — Handle Weight Shape Mismatches (2D → 1D)

Loading pretrained 2D weights into a 1D-adapted model will cause shape mismatches in two key layers:

#### `patch_embed.proj.weight`
- **Pretrained shape:** `(embed_dim, in_chans, tubelet_size, patch_h, patch_w)`
- **New shape:** `(embed_dim, in_chans, tubelet_size, patch_h, 1)`
- **Adaptation:** Average or sum the pretrained 2D spatial kernel across the width dimension:
  ```python
  old_w = checkpoint["module"]["patch_embed.proj.weight"]  # (E, C, T, Ph, Pw)
  new_w = old_w.mean(dim=-1, keepdim=True)                 # (E, C, T, Ph, 1)
  checkpoint["module"]["patch_embed.proj.weight"] = new_w
  ```

#### `pos_embed`
- **Pretrained shape:** Encodes a 2D spatial grid + time.
- **New shape:** 1D sequence + time.
- **Adaptation:** We will interpolate the pre-trained 2D spatial positional embeddings down to the 1D dimension `(512 // 16 = 32)` using `F.interpolate` so that the model retains its learned spatial biases.

### Step 2.3 — Load the Adapted State Dict

```python
# Clean up state dict keys (e.g., removing 'model.encoder.' prefixes if necessary)
state_dict = adapt_checkpoint_for_1d(checkpoint["module"])

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
```

---

## Next Actions

1. Proceed with editing `config.py` to allow tuple inputs.
2. Edit `models_mae_temporal.py` and `pos_embed.py` to handle the asymmetric `(patch_size, 1)` patching and 1D positional embeddings.
3. Write a small test script (`ExampleModelLoading_1D.ipynb` or `.py`) to demonstrate the HuggingFace download, weight interpolation, and successful loading of the `mp_rank_00_model_states.pt` checkpoint.
