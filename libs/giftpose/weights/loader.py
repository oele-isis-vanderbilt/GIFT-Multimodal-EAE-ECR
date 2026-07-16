"""Checkpoint loading for giftpose models.

The runtime consumes legacy mmengine ``.pth`` files directly — the
optimizer / EMA / message-hub blobs they contain are stripped in-memory at
load time. ``mmengine`` / ``mmcv`` / ``mmdet`` do not need to be installed
in the runtime env (a lenient unpickler swaps in a no-op stub for any of
their classes).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

_LEGACY_DROP_PREFIXES = (
    "data_preprocessor.",
    "optimizer.",
    "param_schedulers.",
    "message_hub.",
    "meta.",
)


class _DummyMeta(type):
    """Metaclass: any attribute access on the class returns a no-op callable.

    Pickle sometimes resolves bound methods (e.g.
    ``mmengine.logging.history_buffer.HistoryBuffer.min``) during unpickle —
    a plain stub class without those attrs raises ``AttributeError``. The
    metaclass returns a stub for *anything* asked of it.
    """
    def __getattr__(cls, name: str):
        return _DummyClass


class _DummyClass(metaclass=_DummyMeta):
    """Pickle-shim for arbitrary mmengine/mmdet/mmcv metadata classes that
    appear inside legacy ``.pth`` blobs. We only need their stored
    ``state_dict`` tensors, not the metadata, so an attribute-bag stub
    round-trips fine.
    """

    def __init__(self, *a, **kw): ...

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce__(self):
        return (_DummyClass, ())

    def __call__(self, *a, **kw):
        return _DummyClass()

    def __getattr__(self, name: str):
        return _DummyClass()


def _legacy_torch_load(path: str) -> Any:
    """``torch.load`` for mmengine ``.pth`` checkpoints in environments where
    ``mmengine`` / ``mmcv`` / ``mmdet`` are not installed.

    Patches ``pickle.Unpickler.find_class`` (via ``pickle_module``) so any
    ``mmengine.*`` / ``mmcv.*`` / ``mmdet.*`` class resolves to a stub —
    sufficient to recover the tensor state dict that ``mmengine`` saved.
    """

    class _LenientUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith(("mmengine", "mmcv", "mmdet", "mmpose", "numpy.core")):
                try:
                    return super().find_class(module, name)
                except (AttributeError, ImportError, ModuleNotFoundError):
                    return _DummyClass
            return super().find_class(module, name)

    class _PickleModule:
        Unpickler = _LenientUnpickler

        @staticmethod
        def load(file, **kw):
            return _LenientUnpickler(file, **kw).load()

    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_PickleModule,
    )


def _strip_legacy(state_dict: dict, kind: str) -> dict:
    """Drop optimizer / EMA / meta keys; keep only the prefixes the in-repo
    architecture defines (so ``strict_load`` can verify a clean match).
    """
    if kind == "pose":
        # ``neck.`` covers RTMW/RTMW3D pose models; plain RTMPose checkpoints
        # simply have no neck keys, so the filter is a no-op for them.
        keep = ("backbone.", "neck.", "head.")
    elif kind == "detector":
        keep = ("backbone.", "neck.", "bbox_head.")
    else:
        raise ValueError(f"Unknown kind: {kind!r}")
    out: dict = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in _LEGACY_DROP_PREFIXES):
            continue
        if not any(k.startswith(p) for p in keep):
            continue
        out[k] = v
    return out


def load_state_dict_from_pth(path: str | Path, kind: str) -> dict:
    """Load + clean a giftpose state dict from a legacy mmengine ``.pth`` file.

    ``kind`` selects the prefix filter (``"pose"`` or ``"detector"``).
    """
    p = str(path)
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, ImportError, AttributeError, pickle.UnpicklingError, RuntimeError):
        # mmengine / mmcv / mmdet not installed (ModuleNotFoundError); or
        # weights_only=True / unsafe-globals quirks on older PyTorch
        # (UnpicklingError / RuntimeError) — fall back to lenient unpickle.
        blob = _legacy_torch_load(p)

    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    elif isinstance(blob, dict) and "ema_state_dict" in blob:
        sd = blob["ema_state_dict"]
    elif isinstance(blob, dict):
        sd = blob
    else:
        raise ValueError(f"Unrecognized checkpoint format at {p}")
    return _strip_legacy(sd, kind)


def strict_load(model: torch.nn.Module, state_dict: dict) -> None:
    """Load ``state_dict`` into ``model`` strictly. Reports missing/unexpected
    keys and raises if any drift is detected so weight schema mismatches fail
    fast during dev rather than silently degrading accuracy.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        msg_parts: list[str] = []
        if missing:
            msg_parts.append(
                f"missing keys ({len(missing)}): {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}"
            )
        if unexpected:
            msg_parts.append(
                f"unexpected keys ({len(unexpected)}): {unexpected[:8]}"
                f"{'...' if len(unexpected) > 8 else ''}"
            )
        raise RuntimeError("State-dict mismatch -- " + " | ".join(msg_parts))
