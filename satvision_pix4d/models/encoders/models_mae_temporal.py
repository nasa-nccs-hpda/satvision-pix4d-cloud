from contextlib import nullcontext

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from torch.utils.checkpoint import checkpoint
from satvision_pix4d.models.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid_torch,
)
from satvision_pix4d.models.reconstruction_loss import ReconstructionLoss


def _gather_for_initialization(module):
    """Child layers can already be partitioned inside DeepSpeed zero.Init."""
    parameters = list(module.parameters(recurse=False))
    if any(hasattr(parameter, "ds_id") for parameter in parameters):
        import deepspeed
        return deepspeed.zero.GatheredParameters(parameters, modifier_rank=0)
    return nullcontext()


class PixelRefinementHead(nn.Module):
    def __init__(self, in_chans, hidden_chans=64, depth=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_chans + 1, hidden_chans, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(depth - 2, 0)):
            layers.extend([
                nn.Conv2d(hidden_chans, hidden_chans, kernel_size=3, padding=1),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(hidden_chans, in_chans, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, merged_img, pixel_mask):
        b, t, c, h, w = merged_img.shape
        x = merged_img.reshape(b * t, c, h, w)
        m = pixel_mask.reshape(b * t, 1, h, w).to(x.dtype)
        residual = self.net(torch.cat([x, m], dim=1)).reshape(b, t, c, h, w)
        return merged_img + residual * pixel_mask


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        same_mask=False,
        n_time_components=3,          # e.g. ["year","month","hour"] -> 3
        enc_ts_dim_per_comp=128,      # matches your encoder setting
        dec_ts_dim_per_comp=64,       # matches your decoder setting
        visible_loss_weight=0.0,
        refine_pixels=False,
        refinement_channels=64,
        refinement_depth=3,
        use_checkpoint=False,
        loss_type="mse",
        visnir_channels=(),
        ssim_min=(),
        ssim_max=(),
    ):
        super().__init__()

        if embed_dim % 4 or decoder_embed_dim % 4:
            raise ValueError("Spatial embedding widths must be divisible by 4")
        if embed_dim % num_heads or decoder_embed_dim % decoder_num_heads:
            raise ValueError("Embedding widths must be divisible by attention heads")
        if n_time_components < 1 or enc_ts_dim_per_comp % 2 or dec_ts_dim_per_comp % 2:
            raise ValueError("Temporal components must be positive with even embedding widths")
        if norm_pix_loss and (loss_type != "mse" or refine_pixels):
            raise ValueError("Pixel-space loss/refinement requires norm_pix_loss=False")
        self.use_checkpoint = use_checkpoint
        self.reconstruction_loss = ReconstructionLoss(
            loss_type, in_chans, visnir_channels, ssim_min, ssim_max
        )
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, strict_img_size=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        # timm SDPA dispatches to FlashAttention on supported CUDA hardware/dtypes,
        # with PyTorch's CPU/math fallback. Parameter names stay checkpoint-compatible.
        for block in list(self.blocks) + list(self.decoder_blocks):
            if not hasattr(block.attn, "fused_attn"):
                raise RuntimeError("Install a timm version with PyTorch SDPA attention support")
            block.attn.fused_attn = True

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.same_mask = same_mask
        self.visible_loss_weight = visible_loss_weight
        self.refine_pixels = refine_pixels

        # Encoder projection maps [pos_embed || ts_embed] -> embed_dim
        self.n_time_components = n_time_components
        self.enc_ts_dim_per_comp = enc_ts_dim_per_comp
        self.dec_ts_dim_per_comp = dec_ts_dim_per_comp

        # We don't know grid size at init, but pos_embed_spatial dim = embed_dim (we set it so below),
        # so concat size = embed_dim + n_time_components * enc_ts_dim_per_comp
        self.encoder_temporal_spatial_proj = nn.Linear(
            embed_dim + n_time_components * enc_ts_dim_per_comp, embed_dim
        )

        # Decoder projection maps [pos_embed || ts_embed] -> decoder_embed_dim
        self.decoder_temporal_spatial_proj = nn.Linear(
            decoder_embed_dim + n_time_components * dec_ts_dim_per_comp, decoder_embed_dim
        )
        self.pixel_refinement = (
            PixelRefinementHead(in_chans, refinement_channels, refinement_depth)
            if refine_pixels else None
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        with _gather_for_initialization(self.patch_embed.proj):
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.LayerNorm)):
            with _gather_for_initialization(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    # ---------------- patchify / unpatchify ----------------
    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        B, T, C, H, W = imgs.shape
        if H % p or W % p:
            raise ValueError("Image height and width must be divisible by patch_size")
        h = H // p
        w = W // p
        x = imgs.reshape(B, T, C, h, p, w, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T * h * w, p * p * C)
        return x

    def unpatchify(self, x, T, H, W):
        p = self.patch_embed.patch_size[0]
        B = x.shape[0]
        h = H // p
        w = W // p
        C = x.shape[-1] // (p * p)
        x = x.reshape(B, T, h, w, p, p, C).permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, C, H, W)
        return x

    def tokens_to_pixel_mask(self, mask_tokens, T, H, W):
        p = self.patch_embed.patch_size[0]
        B = mask_tokens.shape[0]
        h = H // p
        w = W // p
        mask = mask_tokens.view(B, T, h, w).unsqueeze(2)
        return mask.repeat_interleave(p, dim=3).repeat_interleave(p, dim=4).float()

    # ---------------- masking ----------------
    def random_masking(self, x, mask_ratio, mask=None):
        N, L, D = x.shape
        if not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be in [0, 1]")
        if mask is not None:
            expected = torch.arange(L, device=x.device).expand(N, -1)
            if (mask.shape != (N, L) or mask.dtype != torch.long
                    or mask.device != x.device
                    or not torch.equal(mask.sort(dim=1).values, expected)):
                raise ValueError("mask must be a LongTensor permutation of token indices per sample")
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) if mask is None else mask
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def same_spatial_masking(self, x, mask_ratio, num_timesteps, patches_per_timestep):
        N, L, D = x.shape
        if not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be in [0, 1]")
        len_keep_spatial = int(patches_per_timestep * (1 - mask_ratio))
        noise = torch.rand(N, patches_per_timestep, device=x.device)
        ids_spatial = torch.argsort(noise, dim=1)
        time_offsets = (
            torch.arange(num_timesteps, device=x.device)
            .view(1, num_timesteps, 1)
            .mul(patches_per_timestep)
        )
        ids_keep = (
            ids_spatial[:, :len_keep_spatial]
            .unsqueeze(1)
            .add(time_offsets)
            .reshape(N, -1)
        )
        ids_remove = (
            ids_spatial[:, len_keep_spatial:]
            .unsqueeze(1)
            .add(time_offsets)
            .reshape(N, -1)
        )
        ids_shuffle = torch.cat([ids_keep, ids_remove], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :ids_keep.shape[1]] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ---------------- temporal embeddings ----------------
    def _compute_temporal_embedding(self, timestamps_flat, embed_dim_per_component):
        ts_embed_list = [
            get_1d_sincos_pos_embed_from_grid_torch(embed_dim_per_component, timestamps_flat[:, k].float())
            for k in range(timestamps_flat.shape[1])
        ]
        return torch.cat(ts_embed_list, dim=1)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (B,T,C,H,W)  z-scored
        pred: (B,L,p*p*C)  ALL tokens in original order (after ids_restore)
        mask: (B,L)        1=masked, 0=visible in original order
        """
        if self.reconstruction_loss.loss_type != "mse":
            B, T, C, H, W = imgs.shape
            return self.reconstruction_loss(
                self.unpatchify(pred.float(), T, H, W), imgs.float(),
                self.tokens_to_pixel_mask(mask, T, H, W), self.visible_loss_weight,
            )
        target = self.patchify(imgs).float()         # (B,L,p*p*C)
        # match precision for mixed precision runs (bf16/fp16/fp32)
        pred = pred.float()
        # optional per-token normalization (like MAE)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        # MSE per token, then average over masked tokens. Optionally add a
        # visible-token term for tiny overfit/debug reconstruction runs.
        loss = (pred - target) ** 2                  # (B,L,p*p*C)
        loss = loss.mean(dim=-1)                     # (B,L)
        masked_denom = mask.sum().clamp_min(1)       # safety
        masked_loss = (loss * mask).sum() / masked_denom
        if self.visible_loss_weight <= 0:
            return masked_loss

        visible = 1.0 - mask
        visible_loss = (loss * visible).sum() / visible.sum().clamp_min(1)
        return masked_loss + self.visible_loss_weight * visible_loss

    def refine_prediction_tokens(self, imgs, pred, mask):
        if self.pixel_refinement is None:
            return pred

        B, T, C, H, W = imgs.shape
        target_tokens = self.patchify(imgs).to(pred.dtype)
        merged_tokens = torch.where(mask.bool().unsqueeze(-1), pred, target_tokens)
        merged_img = self.unpatchify(merged_tokens, T, H, W)
        pixel_mask = self.tokens_to_pixel_mask(mask, T, H, W).to(merged_img.dtype)
        refined_img = self.pixel_refinement(merged_img, pixel_mask)
        refined_tokens = self.patchify(refined_img)
        return torch.where(mask.bool().unsqueeze(-1), refined_tokens, pred)

    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask)
        grid_size = tuple(size // patch for size, patch in
                          zip(imgs.shape[-2:], self.patch_embed.patch_size))
        pred = self.forward_decoder(latent, timestamps, ids_restore, grid_size=grid_size)
        pred = self.refine_prediction_tokens(imgs, pred, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ---------------- forward pass ----------------
    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        if x.ndim != 5:
            raise ValueError("Expected images shaped (B,T,C,H,W)")
        B, T, C, H, W = x.shape
        p = self.patch_embed.patch_size[0]
        if not T or C != self.patch_embed.proj.in_channels or H % p or W % p:
            raise ValueError("Input requires T>0, configured channels, and patch-divisible dimensions")
        if timestamps.shape != (B, T, self.n_time_components):
            raise ValueError(f"Expected timestamps {(B, T, self.n_time_components)}, got {tuple(timestamps.shape)}")
        timestamps = timestamps.to(x.device)

        # patch -> tokens
        # DeepSpeed bf16 weights may run without autocast, while normalization
        # intentionally stays fp32. Cast only the encoder input, preserving the
        # original fp32 images used as reconstruction targets.
        x_flat = x.reshape(B * T, C, H, W).to(dtype=self.patch_embed.proj.weight.dtype)
        x_emb = self.patch_embed(x_flat)                 # (B*T, Ls, embed_dim)
        L_per_step = x_emb.shape[1]
        x_emb = x_emb.reshape(B, T * L_per_step, -1)     # (B, L, embed_dim)

        grid_size = (H // p, W // p)

        # positional (numpy -> torch)
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x_emb.shape[2], grid_size=grid_size, cls_token=False
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device)

        # temporal
        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(
            timestamps_flat, embed_dim_per_component=self.enc_ts_dim_per_comp
        )
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)

        # >>> ensure all inputs match the projection layer's dtype <<<
        proj_dtype = self.encoder_temporal_spatial_proj.weight.dtype
        x_emb = x_emb.to(proj_dtype)
        pos_embed_spatial = pos_embed_spatial.to(proj_dtype)
        ts_embed = ts_embed.to(proj_dtype)

        # concat & project to embed_dim
        embedding = torch.cat([pos_embed_spatial.repeat(T, 1).unsqueeze(0).expand(B, -1, -1), ts_embed], dim=-1)
        embedding = embedding.to(proj_dtype)

        x = x_emb + self.encoder_temporal_spatial_proj(embedding)

        x = x.to(self.cls_token.dtype)
        if self.same_mask and mask is None:
            x, mask, ids_restore = self.same_spatial_masking(x, mask_ratio, T, L_per_step)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if self.use_checkpoint and self.training else blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, timestamps, ids_restore, grid_size=None):
        B = x.shape[0]
        T = timestamps.shape[1]
        L = ids_restore.shape[1]
        L_per_step = L // T

        x = self.decoder_embed(x)

        # reinserting masked tokens (as before)
        mask_tokens = self.mask_token.repeat(B, L + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        if grid_size is None:
            side = int(L_per_step ** 0.5)
            if side * side != L_per_step:
                raise ValueError("Pass grid_size=(height_in_patches, width_in_patches) for rectangular inputs")
            grid_size = (side, side)
        if grid_size[0] * grid_size[1] != L_per_step:
            raise ValueError("grid_size does not match restored tokens")
        timestamps = timestamps.to(x.device)

        # positional for decoder
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x.shape[2], grid_size=grid_size, cls_token=False
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device)
        pos_embed_spatial = pos_embed_spatial.repeat(T, 1)
        pos_embed_spatial = torch.cat([
            torch.zeros(1, pos_embed_spatial.shape[1], device=x.device),
            pos_embed_spatial,
        ], dim=0).unsqueeze(0).expand(B, -1, -1)

        # temporal for decoder
        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(
            timestamps_flat, embed_dim_per_component=self.dec_ts_dim_per_comp
        )
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)
        ts_embed = torch.cat([torch.zeros(B, 1, ts_embed.shape[-1], device=x.device), ts_embed], dim=1)

        # >>> ensure dtype matches decoder projection layer <<<
        proj_dtype = self.decoder_temporal_spatial_proj.weight.dtype
        x = x.to(proj_dtype)
        pos_embed_spatial = pos_embed_spatial.to(proj_dtype)
        ts_embed = ts_embed.to(proj_dtype)

        embedding = torch.cat([pos_embed_spatial, ts_embed], dim=-1).to(proj_dtype)
        x = x + self.decoder_temporal_spatial_proj(embedding)

        x = x.to(self.cls_token.dtype)
        for blk in self.decoder_blocks:
            x = checkpoint(blk, x, use_reentrant=False) if self.use_checkpoint and self.training else blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x
