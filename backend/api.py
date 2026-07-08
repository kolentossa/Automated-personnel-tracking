"""HTTP API routes for the local dashboard and integrations."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.service import AppState


def create_router(state: AppState) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def root() -> dict:
        return {
            "status": "running",
            "device": "RK3588",
            "service": "person tracking",
        }

    @router.get("/status")
    def status() -> dict:
        return state.status()

    @router.get("/events")
    def events(limit: int = Query(default=20, ge=1, le=100)) -> dict:
        return {"events": state.events.recent(limit)}

    @router.get("/statistics")
    def statistics() -> dict:
        return state.statistics()

    return router
