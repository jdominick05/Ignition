# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/assets/__init__.py
Dynamic asset resolver for pre-compiled AIE2 runtime binaries and hardware xclbin overlays.
Provides self-contained discovery via importlib.resources.
"""

from pathlib import Path
import importlib.resources
from typing import Optional


def get_asset_path(filename: str) -> Path:
    """
    Resolves the filesystem path to a bundled AIE2 hardware asset.
    Uses importlib.resources for wheel and package relocatability.
    """
    try:
        traversable = importlib.resources.files("ignition.assets") / filename
        # In Python 3.9+, traversable can be converted to Path if on filesystem
        p = Path(traversable)
        if p.exists():
            return p
    except Exception:
        pass

    # Fallback to local directory relative to this file
    local_p = Path(__file__).parent / filename
    if local_p.exists():
        return local_p

    raise FileNotFoundError(f"Ignition bundled hardware asset '{filename}' could not be located in ignition.assets")


def get_default_xclbin_path() -> Path:
    """Returns the absolute path to the bundled Phoenix AIE2 16-core XCLBIN overlay."""
    return get_asset_path("im2col_4d_16core.xclbin")


def get_transaction_template(name: str = "layer_conv0") -> Path:
    """Returns the path to a bundled CDO transaction binary template."""
    filename = f"{name}.bin" if not name.endswith(".bin") else name
    return get_asset_path(filename)
