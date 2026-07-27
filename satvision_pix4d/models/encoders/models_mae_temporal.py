import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

# These must be in your environment
from satvision_pix4d.models.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid_torch,
)

"""
class FlashMHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, causal):
        super().__init__()
        self.attn = MHA(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        )

    def forward(self, x, attn_mask=None):
        # Always match dtype with the MHA weights
        x = x.to(self.attn.Wqkv.weight.dtype)
        return self.attn(x)


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
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        for blk in self.blocks:
            blk.attn = FlashMHAWrapper(
                embed_dim,
                num_heads=num_heads,
                dropout=0.0,
                causal=False,
            )

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        for blk in self.decoder_blocks:
            blk.attn = FlashMHAWrapper(
                decoder_embed_dim,
                num_heads=decoder_num_heads,
                dropout=0.0,
                causal=False,
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.same_mask = same_mask

        self.encoder_temporal_spatial_proj = None
        self.decoder_temporal_spatial_proj = None

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        B, T, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, T, C, h, p, w, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(B, T * h * w, p * p * C)
        return x

    def unpatchify(self, x, T, H, W):
        p = self.patch_embed.patch_size[0]
        B = x.shape[0]
        L_per_step = x.shape[1] // T
        h = H // p
        w = W // p
        C = x.shape[-1] // (p * p)
        x = x.reshape(B, T, h, w, p, p, C).permute(0, 1, 6, 2, 4, 3, 5).reshape(B, T, C, H, W)
        return x

    def random_masking(self, x, mask_ratio, mask=None):
        N, L, D = x.shape
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

    def _compute_temporal_embedding(self, timestamps_flat, embed_dim_per_component):
        ts_embed_list = [
            get_1d_sincos_pos_embed_from_grid_torch(embed_dim_per_component, timestamps_flat[:, k].float())
            for k in range(timestamps_flat.shape[1])
        ]
        return torch.cat(ts_embed_list, dim=1)

    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        x_emb = self.patch_embed(x_flat)
        L_per_step = x_emb.shape[1]
        x_emb = x_emb.reshape(B, T * L_per_step, -1)

        grid_size = int(L_per_step ** 0.5)
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x_emb.shape[2],
            grid_size=grid_size,
            cls_token=False,
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device).to(x_emb.dtype)
        pos_embed_spatial = pos_embed_spatial.repeat(T, 1).unsqueeze(0).expand(B, -1, -1)

        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(timestamps_flat, embed_dim_per_component=128)
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)

        embedding = torch.cat([pos_embed_spatial, ts_embed], dim=-1)

        if self.encoder_temporal_spatial_proj is None:
            self.encoder_temporal_spatial_proj = nn.Linear(
                embedding.shape[-1],
                x_emb.shape[-1]
            ).to(x.device)

        embedding = self.encoder_temporal_spatial_proj(embedding)
        x = x_emb + embedding

        # Cast to consistent dtype
        x = x.to(self.cls_token.dtype)

        if self.same_mask and mask is None:
            x, mask, ids_restore = self.same_spatial_masking(x, mask_ratio, T, L_per_step)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.blocks:
            x = blk(x.to(self.cls_token.dtype))
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, timestamps, ids_restore):
        B = x.shape[0]
        T = timestamps.shape[1]
        L_per_step = ids_restore.shape[1] // T

        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        grid_size = int(L_per_step ** 0.5)
        pos_embed_spatial = get_2d_sincos_pos_embed(
            embed_dim=x.shape[2],
            grid_size=grid_size,
            cls_token=False,
        )
        pos_embed_spatial = torch.from_numpy(pos_embed_spatial).to(x.device).to(x.dtype)
        pos_embed_spatial = pos_embed_spatial.repeat(T, 1)
        pos_embed_spatial = torch.cat([
            torch.zeros(1, pos_embed_spatial.shape[1], device=x.device),
            pos_embed_spatial,
        ], dim=0).unsqueeze(0).expand(B, -1, -1)

        timestamps_flat = timestamps.reshape(B * T, -1)
        ts_embed = self._compute_temporal_embedding(timestamps_flat, embed_dim_per_component=64)
        ts_embed = ts_embed.reshape(B, T, 1, -1).expand(B, T, L_per_step, -1).reshape(B, T * L_per_step, -1)
        ts_embed = torch.cat([
            torch.zeros(B, 1, ts_embed.shape[-1], device=x.device),
            ts_embed,
        ], dim=1)

        embedding = torch.cat([pos_embed_spatial, ts_embed], dim=-1)

        if self.decoder_temporal_spatial_proj is None:
            self.decoder_temporal_spatial_proj = nn.Linear(
                embedding.shape[-1],
                x.shape[-1]
            ).to(x.device)

        embedding = self.decoder_temporal_spatial_proj(embedding)
        x = x + embedding

        # Cast to consistent dtype
        x = x.to(self.cls_token.dtype)

        for blk in self.decoder_blocks:
            x = blk(x.to(self.cls_token.dtype))
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

from satvision_pix4d.models.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid_torch,
)

class FlashMHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, causal):
        super().__init__()
        if MHA is None:
            raise ImportError(
                "flash_attn is not installed. Use the default timm attention "
                "blocks or install flash-attn in the training environment."
            )
        self.attn = MHA(embed_dim, num_heads=num_heads, dropout=dropout, causal=causal)

    def forward(self, x, attn_mask=None):
        x = x.to(self.attn.Wqkv.weight.dtype)
        return self.attn(x)


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
        # NEW: make temporal embedding sizing explicit
        n_time_components=3,          # e.g. ["year","month","hour"] -> 3
        enc_ts_dim_per_comp=128,      # matches your encoder setting
        dec_ts_dim_per_comp=64,       # matches your decoder setting
        visible_loss_weight=0.0,
        refine_pixels=False,
        refinement_channels=64,
        refinement_depth=3,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
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

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.same_mask = same_mask
        self.visible_loss_weight = visible_loss_weight
        self.refine_pixels = refine_pixels

        # ---------- NEW: register temporal+spatial projection layers here ----------
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
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ---------------- patchify / unpatchify ----------------
    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        B, T, C, H, W = imgs.shape
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
        target = self.patchify(imgs)                 # (B,L,p*p*C)
        # match precision for mixed precision runs (bf16/fp16/fp32)
        target = target.to(pred.dtype)
        # optional per-token normalization (like MAE)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
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
        return self.patchify(refined_img)

    def forward(self, imgs, timestamps, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, mask_ratio, mask)
        pred = self.forward_decoder(latent, timestamps, ids_restore)
        pred = self.refine_prediction_tokens(imgs, pred, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ---------------- forward pass ----------------
    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        B, T, C, H, W = x.shape

        # patch -> tokens
        x_flat = x.reshape(B * T, C, H, W)
        x_emb = self.patch_embed(x_flat)                 # (B*T, Ls, embed_dim)
        L_per_step = x_emb.shape[1]
        x_emb = x_emb.reshape(B, T * L_per_step, -1)     # (B, L, embed_dim)

        grid_size = int(L_per_step ** 0.5)

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

        # keep the rest as-is
        x = x.to(self.cls_token.dtype)
        if self.same_mask and mask is None:
            x, mask, ids_restore = self.same_spatial_masking(x, mask_ratio, T, L_per_step)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        for blk in self.blocks:
            x = blk(x.to(self.cls_token.dtype))
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, timestamps, ids_restore):
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

        grid_size = int(L_per_step ** 0.5)

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
            x = blk(x.to(self.cls_token.dtype))
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x



# Test
if __name__ == "__main__":
    B, T, C, H, W = 2, 3, 14, 512, 512
    TT = 5

    imgs = torch.randn(B, T, C, H, W)
    timestamps = torch.randint(low=0, high=1000, size=(B, T, TT))

    model = MaskedAutoencoderViT(
        img_size=H,
        patch_size=16,
        in_chans=C,
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    imgs = imgs.to(device)
    timestamps = timestamps.to(device)

    loss, pred, mask = model(imgs, timestamps)
    print("Loss:", loss.item())
    print("Pred shape:", pred.shape)
