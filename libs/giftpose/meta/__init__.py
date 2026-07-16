import importlib
from types import ModuleType

from libs.giftpose.meta.halpe26 import (
    KEYPOINT_NAMES,
    FLIP_INDICES,
    SIGMAS,
    SKELETON,
    NUM_KEYPOINTS,
)

__all__ = [
    "KEYPOINT_NAMES", "FLIP_INDICES", "SIGMAS", "SKELETON", "NUM_KEYPOINTS",
    "get_meta",
]


def get_meta(name: str) -> ModuleType:
    """Return the keypoint-metadata module for a dataset name.

    ``name`` matches :attr:`libs.giftpose.registry.ArchSpec.meta` (e.g.
    ``"halpe26"``, ``"wholebody133"``). Each module exposes
    ``NUM_KEYPOINTS``, ``KEYPOINT_NAMES``, ``FLIP_INDICES``, ``SIGMAS``
    and ``SKELETON``.
    """
    return importlib.import_module(f"libs.giftpose.meta.{name}")
