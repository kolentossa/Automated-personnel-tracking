"""FastAPI entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import create_router
from backend.service import AppState, PROJECT_ROOT

app = FastAPI(title="Automated Personnel Tracking System", version="0.1.0")
state = AppState(PROJECT_ROOT)
app.include_router(create_router(state))

frontend_dir = Path(PROJECT_ROOT) / "frontend"
if frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_dir), html=True), name="dashboard")


@app.on_event("startup")
def startup() -> None:
    state.start_from_env()


@app.on_event("shutdown")
def shutdown() -> None:
    state.stop()
