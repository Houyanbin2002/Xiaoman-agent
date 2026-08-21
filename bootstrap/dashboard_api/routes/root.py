from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response


def register_root_route(app: FastAPI, *, static_dir: Path) -> None:
    @app.get("/")
    def dashboard_index() -> Response:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return Response(
                content="Dashboard 前端尚未构建，请先运行 `npm run build`。",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        html = index_file.read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")
