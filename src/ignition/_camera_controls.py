# Copyright (C) 2026 The Ignition contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignition/_camera_controls.py
A UVC webcam's exposure auto priority, through DirectShow's IKsPropertySet (ctypes COM, Windows only).

With priority 1, auto exposure may lower the frame rate to lengthen exposure in dim light; with 0 it holds the
frame rate and the image comes out darker. The setting belongs to the camera and outlives the process, so a caller
that changes it puts back the value it read. Device indices follow DirectShow's video input enumeration, the
order OpenCV's CAP_DSHOW indices use.
"""

import ctypes
from ctypes import POINTER, byref, c_long, c_ulong, c_void_p, wintypes
from typing import List

AUTO_EXPOSURE_PRIORITY = 19  # KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY
_FLAGS_MANUAL = 0x2
_VT_BSTR = 8
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: COM already initialised on this thread in another mode


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class _VARIANT(ctypes.Structure):
    _fields_ = [("vt", wintypes.WORD), ("r1", wintypes.WORD), ("r2", wintypes.WORD), ("r3", wintypes.WORD),
                ("val", c_void_p), ("pad", c_void_p)]


class _KSIDENTIFIER(ctypes.Structure):
    _fields_ = [("Set", _GUID), ("Id", c_ulong), ("Flags", c_ulong)]


class _KSPROPERTY(ctypes.Union):
    # KSIDENTIFIER is a union with a LONGLONG: its 8-byte alignment makes KSPROPERTY_CAMERACONTROL_S 40 bytes on
    # x64, and the driver rejects a 36-byte buffer with ERROR_INSUFFICIENT_BUFFER.
    _fields_ = [("s", _KSIDENTIFIER), ("Alignment", ctypes.c_longlong)]


class _KSPROPERTY_CAMERACONTROL_S(ctypes.Structure):
    _fields_ = [("Property", _KSPROPERTY), ("Value", c_long), ("Flags", c_ulong), ("Capabilities", c_ulong)]


_ole32 = ctypes.WinDLL("ole32")
_oleaut32 = ctypes.WinDLL("oleaut32")
_ole32.CLSIDFromString.argtypes = [wintypes.LPCWSTR, POINTER(_GUID)]
_ole32.CLSIDFromString.restype = c_long
_ole32.CoInitializeEx.argtypes = [c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = c_long
_ole32.CoUninitialize.argtypes = []
_ole32.CoUninitialize.restype = None
_ole32.CoCreateInstance.argtypes = [POINTER(_GUID), c_void_p, wintypes.DWORD, POINTER(_GUID), POINTER(c_void_p)]
_ole32.CoCreateInstance.restype = c_long
_oleaut32.VariantClear.argtypes = [POINTER(_VARIANT)]


def _check(hr: int, what: str) -> int:
    if hr < 0:
        raise OSError(f"{what} failed: HRESULT 0x{hr & 0xFFFFFFFF:08X}")
    return hr


def _guid(text: str) -> _GUID:
    g = _GUID()
    _check(_ole32.CLSIDFromString(text, byref(g)), f"CLSIDFromString {text}")
    return g


def _method(ptr: c_void_p, index: int, *argtypes):
    """COM method ``index`` of the interface at ``ptr``; call it with ``ptr`` as the first argument."""
    vtbl = ctypes.cast(ctypes.cast(ptr, POINTER(c_void_p))[0], POINTER(c_void_p))
    return ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)(vtbl[index])


def _release(ptr: c_void_p) -> None:
    if ptr:
        _method(ptr, 2)(ptr)


_CLSID_SystemDeviceEnum = _guid("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
_IID_ICreateDevEnum = _guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
_CLSID_VideoInputDeviceCategory = _guid("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
_IID_IBaseFilter = _guid("{56a86895-0ad4-11ce-b03a-0020af0ba770}")
_IID_IPropertyBag = _guid("{55272A00-42CB-11CE-8135-00AA004BB851}")
_IID_IKsPropertySet = _guid("{31EFAC30-515C-11d0-A9AA-00AA0061BE93}")
_PROPSETID_VIDCAP_CAMERACONTROL = _guid("{C6E13370-30AC-11d0-A18C-00A0C9118956}")


class ExposurePriority:
    """Exposure auto priority of DirectShow video input device ``index``; use as a context manager."""

    def __init__(self, index: int):
        self.index = index
        self.name = f"camera {index}"
        self._monikers: List[c_void_p] = []
        self._filter = c_void_p()
        self._ks = c_void_p()
        hr = _ole32.CoInitializeEx(None, 0)
        self._uninitialize = hr in (0, 1)
        if hr != _RPC_E_CHANGED_MODE:
            _check(hr, "CoInitializeEx")
        try:
            self._open()
        except Exception:
            self.close()
            raise

    def _open(self) -> None:
        dev_enum, enum = c_void_p(), c_void_p()
        _check(_ole32.CoCreateInstance(byref(_CLSID_SystemDeviceEnum), None, 1, byref(_IID_ICreateDevEnum),
                                       byref(dev_enum)), "CoCreateInstance(SystemDeviceEnum)")
        try:
            hr = _method(dev_enum, 3, POINTER(_GUID), POINTER(c_void_p), wintypes.DWORD)(
                dev_enum, byref(_CLSID_VideoInputDeviceCategory), byref(enum), 0)
            _check(hr, "CreateClassEnumerator")
            if hr == 0 and enum:
                while True:
                    moniker, fetched = c_void_p(), c_ulong()
                    hr = _method(enum, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))(
                        enum, 1, byref(moniker), byref(fetched))
                    if hr != 0 or fetched.value == 0:
                        break
                    self._monikers.append(moniker)
                _release(enum)
        finally:
            _release(dev_enum)
        if self.index >= len(self._monikers):
            raise OSError(f"no DirectShow video input device {self.index} ({len(self._monikers)} present)")
        moniker = self._monikers[self.index]
        self.name = self._friendly_name(moniker)
        _check(_method(moniker, 8, c_void_p, c_void_p, POINTER(_GUID), POINTER(c_void_p))(
            moniker, None, None, byref(_IID_IBaseFilter), byref(self._filter)), "BindToObject(IBaseFilter)")
        _check(_method(self._filter, 0, POINTER(_GUID), POINTER(c_void_p))(
            self._filter, byref(_IID_IKsPropertySet), byref(self._ks)), "QueryInterface(IKsPropertySet)")
        support = wintypes.DWORD()
        hr = _method(self._ks, 5, POINTER(_GUID), wintypes.DWORD, POINTER(wintypes.DWORD))(
            self._ks, byref(_PROPSETID_VIDCAP_CAMERACONTROL), AUTO_EXPOSURE_PRIORITY, byref(support))
        if hr != 0 or (support.value & 3) != 3:  # KSPROPERTY_SUPPORT_GET | KSPROPERTY_SUPPORT_SET
            raise OSError(f"{self.name} does not let exposure auto priority be read and set")

    @staticmethod
    def _friendly_name(moniker: c_void_p) -> str:
        bag = c_void_p()
        hr = _method(moniker, 9, c_void_p, c_void_p, POINTER(_GUID), POINTER(c_void_p))(
            moniker, None, None, byref(_IID_IPropertyBag), byref(bag))
        if hr != 0:
            return "camera"
        try:
            var = _VARIANT()
            hr = _method(bag, 3, wintypes.LPCWSTR, POINTER(_VARIANT), c_void_p)(bag, "FriendlyName", byref(var), None)
            name = ctypes.wstring_at(var.val) if hr == 0 and var.vt == _VT_BSTR and var.val else "camera"
            _oleaut32.VariantClear(byref(var))
            return name
        finally:
            _release(bag)

    def _call(self, index: int, prop: _KSPROPERTY_CAMERACONTROL_S, what: str) -> None:
        header = ctypes.sizeof(_KSPROPERTY)
        base = ctypes.addressof(prop)
        args = [byref(_PROPSETID_VIDCAP_CAMERACONTROL), AUTO_EXPOSURE_PRIORITY, base + header,
                ctypes.sizeof(prop) - header, base, ctypes.sizeof(prop)]
        argtypes = [POINTER(_GUID), wintypes.DWORD, c_void_p, wintypes.DWORD, c_void_p, wintypes.DWORD]
        if index == 4:  # Get also returns the byte count
            returned = wintypes.DWORD()
            args.append(byref(returned))
            argtypes.append(POINTER(wintypes.DWORD))
        _check(_method(self._ks, index, *argtypes)(self._ks, *args), what)

    def get(self) -> int:
        prop = _KSPROPERTY_CAMERACONTROL_S()
        self._call(4, prop, "IKsPropertySet::Get(exposure auto priority)")
        return int(prop.Value)

    def set(self, value: int) -> None:
        prop = _KSPROPERTY_CAMERACONTROL_S()
        prop.Value, prop.Flags = int(value), _FLAGS_MANUAL
        self._call(3, prop, "IKsPropertySet::Set(exposure auto priority)")

    def close(self) -> None:
        _release(self._ks)
        _release(self._filter)
        for moniker in self._monikers:
            _release(moniker)
        self._ks, self._filter, self._monikers = c_void_p(), c_void_p(), []
        if self._uninitialize:
            _ole32.CoUninitialize()
            self._uninitialize = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
