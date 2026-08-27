"""FastAPI application for the analysis viewer sidecar."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config as cfg
from .routers import adjustments, compare, media, runs, session, sidecars


def create_app() -> FastAPI:
    app = FastAPI(
        title="Analysis Viewer Sidecar",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        # Keep `/compare` and `/compare/` (and `/analysis`) independent: the data
        # API lives under /api/*, while the bare paths serve the SPA. Without this,
        # Starlette would 307-redirect one slash variant into the other.
        redirect_slashes=False,
    )

    # Desktop (Tauri) origins by default. In `--serve` mode the UI is served
    # same-origin from this backend, so CORS is moot; extra origins can still be
    # allowed via GIFT_VIEWER_ALLOWED_ORIGINS (comma-separated) for dev setups.
    origins = [
        "tauri://localhost",
        "https://tauri.localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra = os.environ.get("GIFT_VIEWER_ALLOWED_ORIGINS", "")
    origins += [o.strip() for o in extra.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Range", "Content-Type"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    # All data endpoints live under /api/* so the bare `/analysis` and `/compare`
    # paths are free to serve the SPA shell (the comparison data route is now
    # /api/compare, not /compare).
    app.include_router(session.router, prefix="/api")
    app.include_router(sidecars.router, prefix="/api")
    app.include_router(media.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(compare.router, prefix="/api")
    app.include_router(adjustments.router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "analysis-viewer-sidecar"}

    @app.get("/api/config")
    def get_config() -> dict:
        """Runtime config the frontend reads at startup. In server mode it learns
        the outputs root so Compare can browse every run under it."""
        root = cfg.outputs_root()
        return {"outputs_root": root, "serve_mode": root is not None}

    # In server mode, serve the built Vue app from the same origin so a single URL
    # works from any device on the network. The SPA is a single page; the path
    # before `?run=` selects the initial view (Analysis vs Compare), parsed in
    # App.vue. Serve index.html for each known view path so a deep link boots the
    # app; both slash variants are registered because redirect_slashes is off.
    dist = cfg.dist_dir()
    if dist:
        index_file = os.path.join(dist, "index.html")

        def spa() -> FileResponse:
            return FileResponse(index_file)

        for path in ("/", "/analysis", "/analysis/", "/compare", "/compare/"):
            app.add_api_route(path, spa, methods=["GET"], include_in_schema=False)

        # Mounted LAST so the /api/* routes and the SPA paths above take
        # precedence; this serves /assets/* and other static files. Keep it last.
        app.mount("/", StaticFiles(directory=dist, html=True), name="static")

    return app


app = create_app()
