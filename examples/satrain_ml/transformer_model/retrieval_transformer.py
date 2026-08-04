"""Custom temporal-aware retrieval callback for IPWGTransformer.

Mirrors satrain.pytorch.PytorchRetrieval.__call__'s conventions (dim detection, feature_dim
semantics, `.select(feature_dim, 0)` indexing, sigmoid-applied-in-callback for logits,
precip_flag/heavy_precip_flag thresholding, output xr.Dataset construction) but is temporal-aware:
it reshapes `obs_geo` (features=112) into the time-major `(B,7,16,H,W)` layout the model expects,
and builds `static = cat([obs_gmi, ancillary])` itself instead of relying on
`PytorchRetrieval(stack=True)` (which would concatenate everything, including geo_t, into one
undifferentiated feature axis and destroy the (T,C) structure).
"""
import torch
import xarray as xr

from model_transformer import N_TIMESTEPS, N_ABI_CH

N_GEO_FEATURES = N_TIMESTEPS * N_ABI_CH  # 112, time-major: flat index = t * 16 + c


def make_temporal_retrieval_fn(
    model,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    precip_threshold: float = 0.5,
    heavy_precip_threshold: float = 0.5,
    logits: bool = True,
):
    """Builds a `retrieval_fn(input_data: xr.Dataset) -> xr.Dataset` closure bound to `model`."""
    model = model.to(device=device).eval()

    def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
        if "scan" in input_data.dims:
            spatial_dims = ("scan", "pixel")
        elif "latitude" in input_data.dims:
            spatial_dims = ("latitude", "longitude")
        else:
            spatial_dims = ()

        assert "batch" in input_data.dims, (
            "custom temporal retrieval_fn requires a batched xr.Dataset — pass batch_size to "
            "Evaluator.evaluate (the model's per-timestep indexing assumes a leading batch axis)"
        )
        dims = ("batch",) + spatial_dims
        feature_dim = 1

        obs_geo = torch.tensor(input_data["obs_geo"].data).to(device, dtype)
        obs_gmi = torch.tensor(input_data["obs_gmi"].data).to(device, dtype)
        ancillary = torch.tensor(input_data["ancillary"].data).to(device, dtype)

        n_geo_features = obs_geo.shape[feature_dim]
        assert n_geo_features == N_GEO_FEATURES, (
            f"obs_geo feature dim must be {N_GEO_FEATURES} ({N_TIMESTEPS}*{N_ABI_CH}); "
            f"got {n_geo_features}. Did the eval `inputs` list use GeoT (not Geo)?"
        )

        # Time-major reshape: flat index t*16+c -> (T,C). NEVER (C,T) — see DESIGN_NOTES.md pitfall #1.
        new_shape = list(obs_geo.shape)
        new_shape[feature_dim:feature_dim + 1] = [N_TIMESTEPS, N_ABI_CH]
        geo_t = obs_geo.reshape(new_shape)  # (B, 7, 16, H, W)

        static = torch.cat([obs_gmi, ancillary], dim=feature_dim)  # (B, 19, H, W)

        with torch.no_grad():
            pred = model(geo_t, static)

        results = xr.Dataset()

        sp = pred["surface_precip"].select(feature_dim, 0)
        # Model trained on log1p(mm/h) (matches model_transformer.py's active _compute_losses
        # and the baseline's final v4log1p variant) — expm1 back to mm/h. Disabled hook, only
        # for a raw-MSE-trained checkpoint variant (must match model_transformer.py's commented
        # OLD COMPUTE LOSSES branch if ever enabled):
        # sp = sp
        sp = torch.expm1(sp.clamp(min=0))
        results["surface_precip"] = (dims, sp.float().cpu().numpy())

        pop = pred["probability_of_precip"].select(feature_dim, 0)
        if logits:
            pop = torch.sigmoid(pop)
        pop = pop.float().cpu().numpy()
        results["probability_of_precip"] = (dims, pop)
        results["precip_flag"] = (dims, precip_threshold <= pop)

        pohp = pred["probability_of_heavy_precip"].select(feature_dim, 0)
        if logits:
            pohp = torch.sigmoid(pohp)
        pohp = pohp.float().cpu().numpy()
        results["probability_of_heavy_precip"] = (dims, pohp)
        results["heavy_precip_flag"] = (dims, heavy_precip_threshold <= pohp)

        return results

    return retrieval_fn
