"""Official-checkpoint auto-download (stdlib only — no new dependencies).

Resolution contract (see ``resolve_weights``): an explicitly configured,
existing weights file always wins; otherwise, when auto-download is allowed
and the architecture spec carries an official URL, the checkpoint is fetched
once into the models directory (atomic rename, size-checked) and reused on
every later run. The project's fine-tuned default weights are never
downloaded — they have no URL in the registry.
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def download_checkpoint(url: str, dest_path: str, expected_mb: int = 0) -> str:
    """Download ``url`` to ``dest_path`` (atomic). Returns ``dest_path``.

    Writes to a temp file in the destination directory first and renames on
    success, so an interrupted download never leaves a truncated ``.pth``
    that a later run would try to load.
    """
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading %s -> %s (~%d MB)", url, dest_path, expected_mb)

    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total and got % (64 * _CHUNK) < _CHUNK:
                    logger.info("  %d/%d MB", got >> 20, total >> 20)
        if total and got != total:
            raise IOError(
                f"Incomplete download for {url}: got {got} of {total} bytes"
            )
        os.replace(tmp_path, dest_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Downloaded %s (%d MB)", dest_path, os.path.getsize(dest_path) >> 20)
    return dest_path


def resolve_weights(
    spec,
    configured_path: str | None,
    models_dir: str = "models",
    auto_download: bool = False,
) -> str:
    """Return a usable weights path for ``spec``.

    Order:
      1. ``configured_path`` if it exists on disk (explicit config wins).
      2. The registry cache file ``<models_dir>/<spec.default_filename>``
         if already downloaded.
      3. Auto-download from ``spec.url`` when ``auto_download`` is true.
      4. Otherwise raise with the exact URL so the user can fetch manually.
    """
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    cache_path = (
        os.path.join(models_dir, spec.default_filename)
        if spec.default_filename else None
    )
    if cache_path and os.path.isfile(cache_path):
        return cache_path

    if spec.url and cache_path:
        if auto_download:
            return download_checkpoint(spec.url, cache_path, spec.approx_mb)
        raise FileNotFoundError(
            f"Weights for {spec.tag!r} not found "
            f"(looked for {configured_path!r} and {cache_path!r}). "
            f"Enable auto-download with \"auto_download_models\": true in the "
            f"config, or download manually:\n  curl -L -o {cache_path} {spec.url}"
        )
    raise FileNotFoundError(
        f"Weights for {spec.tag!r} not found at {configured_path!r} and the "
        f"registry has no official checkpoint URL for this tag (fine-tuned "
        f"project weights must be provided explicitly)."
    )
