# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/devices.py
NPU discovery through pyxrt. Every field is what XRT reports for the device; nothing is filled in.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

_log = logging.getLogger("ignition")


@dataclass
class DeviceInfo:
    """An AMD XDNA NPU as pyxrt reports it. A value XRT does not return is ``None``."""
    device_id: int
    name: str
    bdf: str
    xrt_version: Optional[str]
    driver_version: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "bdf": self.bdf,
            "xrt_version": self.xrt_version,
            "driver_version": self.driver_version,
        }

    def __str__(self) -> str:
        return (
            f"Device {self.device_id}: {self.name} [{self.bdf}]\n"
            f"  XRT:         {self.xrt_version or 'not reported'}\n"
            f"  NPU driver:  {self.driver_version or 'not reported'}"
        )


def _host_versions(dev: Any, info: Any) -> Tuple[Optional[str], Optional[str]]:
    """(XRT version, NPU driver version) from XRT's ``host`` report; ``None`` where XRT gives no value."""
    try:
        host = json.loads(dev.get_info(info.host))
        drivers = host.get("drivers") or []
        npu = [d for d in drivers if "NPU" in str(d.get("name", ""))]
        driver = (npu or drivers or [{}])[0]
        return host.get("version"), driver.get("version")
    except Exception as exc:  # noqa: BLE001 - the versions are optional
        _log.debug("[Ignition] pyxrt host report unavailable: %s: %s", type(exc).__name__, exc)
        return None, None


def probe_devices() -> List[DeviceInfo]:
    """
    Probes NPU Device 0 through pyxrt.
    Returns a one-element list when XRT opens it, otherwise an empty list.
    """
    try:
        from ignite_xdna.runtime.driver import setup_xrt_environment
        setup_xrt_environment()
        import pyxrt

        dev = pyxrt.device(0)
        info = pyxrt.xrt_info_device
        xrt_version, driver_version = _host_versions(dev, info)
        return [
            DeviceInfo(
                device_id=0,
                name=dev.get_info(info.name),
                bdf=dev.get_info(info.bdf),
                xrt_version=xrt_version,
                driver_version=driver_version,
            )
        ]
    except Exception as exc:  # noqa: BLE001 - no pyxrt, no driver or no NPU all mean "none detected"
        _log.debug("[Ignition] pyxrt found no NPU: %s: %s", type(exc).__name__, exc)
        return []


def get_default_device() -> Optional[DeviceInfo]:
    """Returns the primary detected AMD XDNA NPU, or None if none detected."""
    devs = probe_devices()
    return devs[0] if devs else None
