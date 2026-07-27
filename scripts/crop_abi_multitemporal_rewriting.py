#!/usr/bin/env python3
"""Compatibility entry point for the CloudSat-ABI collocation CLI."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from satvision_pix4d.view.cloudsat_abi_cropping_cli import main


if __name__ == "__main__":
    sys.exit(main())
