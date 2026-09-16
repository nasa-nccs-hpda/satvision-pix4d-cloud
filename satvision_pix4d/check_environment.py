"""Check the uv training environment before allocating a foundation model."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--require-deepspeed', action='store_true')
    args = parser.parse_args(argv)
    report = {name: version(name) for name in ['torch', 'torchvision', 'lightning', 'timm', 'tensorboard', 'yacs']}
    # Check actual imports too: versions alone cannot detect ABI/import failures.
    from satvision_pix4d.benchmark import SyntheticMAEBenchmark  # noqa: F401
    from satvision_pix4d.satvision_pix4d_cli import main as train_main  # noqa: F401
    if args.require_deepspeed:
        # DeepSpeed 0.17.6 probes nvcc for optional op compatibility on import,
        # even when installation used DS_BUILD_OPS=0.
        if torch.cuda.is_available():
            from torch.utils.cpp_extension import CUDA_HOME
            nvcc = Path(CUDA_HOME) / 'bin' / 'nvcc' if CUDA_HOME else None
            if nvcc is None or not os.access(nvcc, os.X_OK):
                parser.error(
                    'DeepSpeed 0.17.6 requires a discoverable CUDA toolkit on GPU nodes. '
                    'Load your site CUDA module or set CUDA_HOME to the real toolkit root '
                    '(containing bin/nvcc), then retry in a fresh Python process. '
                    'For cu128 use CUDA toolkit 12.8. nvidia-smi reports driver support, '
                    'not toolkit installation. See docs/uv-environment.md.'
                )
            report['cuda_toolkit'] = str(CUDA_HOME)
        import deepspeed
        report['deepspeed'] = deepspeed.__version__
    report['cuda_runtime'] = torch.version.cuda
    report['devices'] = []
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Run on an allocated GPU node with the cu126/cu128 extra and a compatible driver.')
        devices = [torch.device('cuda', i) for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device('cpu')]
    for device in devices:
        if device.type == 'cuda':
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f'{device} does not support the configured bf16 precision')
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        q = torch.randn(1, 4, 128, 64, device=device, dtype=dtype, requires_grad=True)
        if device.type == 'cuda':
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = F.scaled_dot_product_attention(q, q, q)
        else:
            result = F.scaled_dot_product_attention(q, q, q)
        result.float().square().mean().backward()
        if not torch.isfinite(result).all() or not torch.isfinite(q.grad).all():
            raise RuntimeError(f'Attention forward/backward produced non-finite values on {device}')
        report['devices'].append({
            'device': str(device),
            'name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'attention_forward_backward': 'passed',
            'backend': 'flash' if device.type == 'cuda' else 'automatic',
        })
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
