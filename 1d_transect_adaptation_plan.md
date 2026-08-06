# Handoff Plan: Adapt satvision-pix4d-base (SatMAE ViT) for 1D CloudSat/ABI Transects

## Background & Architecture Strategy

The preprocessing pipeline now produces **1D transect/profile** data shaped `512×1×16×7` (512 CloudSat footprints × 1 nearest ABI pixel × 16 channels × 7 timesteps), instead of the previous 2D spatial tiles (`512×512×16×7`).

- **Architecture:** The `satvision-pix4d-base` model is a **Masked Autoencoder built on a ViT** (SatMAE). We will reuse the existing `MaskedAutoencoderViTTemporal` code in the repository. 
- **HuggingFace Checkpoint:** The weights are stored in `mp_rank_00_model_states.pt` on a gated HuggingFace repository (`nasa-cisto-data-science-group/satvision-pix4d-base`).
- **Temporal Dimension:** The 7 timesteps will not be compressed into channels. The MAE encoder will process the spatio-temporal tokens.

---

## Phase 1: Modify the Base Architecture & Config for 1D `512×1` Input (COMPLETED)

### Step 1.1 — Update Configuration Typing in `config.py`

**File:** `satvision_pix4d/configs/config.py`

**Problem:** `yacs` enforces strict type matching. Overriding scalar integer defaults (e.g. `224`) with `[512, 1]` triggers a `TypeError`.

**Change:** Convert defaults to tuples/lists so YAML overrides like `[512, 1]` are accepted:
```python
_C.DATA.IMG_SIZE = (224, 224)
_C.MODEL.MAE_VIT.PATCH_SIZE = (16, 16)
```

### Step 1.2 — Adapt Patch Embedding for Asymmetric (1D) Input

**File:** `satvision_pix4d/models/encoders/models_mae_temporal.py`

The standard `PatchEmbed` expects square patches. For a `512×1` input, this must become asymmetric.
Modify the `PatchEmbed` implementation to support tuple patch sizes (e.g., `(16, 1)`):
```python
# The patch embedding kernel and stride will become:
nn.Conv3d(in_chans, embed_dim, kernel_size=(tubelet_size, patch_h, 1), stride=(tubelet_size, patch_h, 1))
```

### Step 1.3 — Adapt Positional Embeddings for 1D Spatial + Temporal

**File:** `satvision_pix4d/models/utils/pos_embed.py`

The existing `get_2d_sincos_pos_embed` creates a 2D grid. For a `512×1` input, this degenerates to a 1D sequence in space.
Add a conditional path:
- If the spatial width is `1` (i.e., 1D transect), use `get_1d_sincos_pos_embed_from_grid_torch` for the spatial dimension.
- Combine this with the existing 1D temporal positional embedding to create the final 3D (Time + Space 1D) positional embedding.

---

## Phase 2: Load Pretrained Weights from HuggingFace (COMPLETED)

### Step 2.1 — Download Weights via `huggingface_hub`

Since the repository is gated, we will require HuggingFace authentication. We will write a loading script that uses `token=True`:

```python
from huggingface_hub import hf_hub_download

# Uses the cached HF token (run `huggingface-cli login` first)
model_path = hf_hub_download(
    repo_id="nasa-cisto-data-science-group/satvision-pix4d-base",
    filename="mp_rank_00_model_states.pt",
    token=True 
)
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
```

### Step 2.2 — Handle Weight Shape Mismatches (2D → 1D)

Loading pretrained 2D weights into a 1D-adapted model will cause shape mismatches:

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
- **Adaptation:** Interpolate the pre-trained 2D spatial positional embeddings down to the 1D dimension `(512 // 16 = 32)` using `F.interpolate` so that the model retains its learned spatial biases.

### Step 2.3 — Load the Adapted State Dict

```python
state_dict = adapt_checkpoint_for_1d(checkpoint["module"])
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
```

---

## Phase 3: Finetuning & Data Loading (COMPLETED)

The next session will focus on finetuning the model utilizing existing logic from the `dev` branch. Reference the following notebooks for the blueprint:
- `notebooks/cloud_height/CloudHeight_SatVision-Pix4D_pair.ipynb`
- `notebooks/cloud_height/CloudHeight_SatVision-Pix4D.ipynb`

As established: *"the workflow is the same: load pre-trained SatMAE as encoder, build customized decoder, then load label data to fine tune."*

### Tasks for Next Session:
1. **Dataset Classes:** Adapt the PyTorch Dataset classes found in the referenced notebooks to parse the new `512x1x16x7` spatio-temporal inputs.
2. **Custom Decoder:** Port and modify the customized decoders from the notebooks to process the 1D feature outputs from the adapted SatMAE ViT encoder and map them to a 1D target output (e.g. cloud properties along the transect).
3. **Training Loop:** Initialize the finetuning loop using the interpolated/reshaped pre-trained weights to train the downstream task.
