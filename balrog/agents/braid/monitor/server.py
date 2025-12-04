"""Starlette web server for BRAID monitor."""

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from .queries import MonitorDB

MONITOR_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=MONITOR_DIR / "templates")


def create_app(db_path: Path) -> Starlette:
    """Create Starlette app with routes."""
    db = MonitorDB(db_path)

    async def index(request: Request) -> HTMLResponse:
        agents = db.get_all_agents()
        selected = request.query_params.get("agent")
        if not selected and agents:
            selected = agents[0].worker_id
        return templates.TemplateResponse(
            request, "index.html", {"agents": agents, "selected": selected}
        )

    async def partials_agents(request: Request) -> HTMLResponse:
        agents = db.get_all_agents()
        selected = request.query_params.get("agent", "")
        return templates.TemplateResponse(
            request, "partials/agents.html", {"agents": agents, "selected": selected}
        )

    async def partials_all(request: Request) -> HTMLResponse:
        worker_id = request.path_params["worker_id"]

        # Step navigation
        max_step = db.get_max_step(worker_id)
        step_param = request.query_params.get("step")
        if step_param and step_param.isdigit():
            current_step = min(int(step_param), max_step)
        else:
            current_step = None  # None means "latest"

        screen = db.get_latest_screen(worker_id, max_step=current_step)
        stats = db.get_stats(worker_id)
        responses = db.get_recent_responses(worker_id, limit=10, max_step=current_step)

        # Memory filters from query params
        scope = request.query_params.get("scope")
        if scope not in ("persistent", "episode"):
            scope = None
        include_deleted = request.query_params.get("deleted") == "1"

        # Pass current episode for episode-scoped memory filtering
        current_episode = stats.episode if stats else None
        entries = db.get_memory_entries(
            limit=50, scope=scope, include_deleted=include_deleted, episode=current_episode
        )

        # Get full prompt and response for debug panel
        full_prompt = db.get_latest_prompt(worker_id, max_step=current_step)
        full_response = db.get_latest_full_response(worker_id, max_step=current_step)

        return templates.TemplateResponse(
            request,
            "partials/all.html",
            {
                "screen": screen,
                "stats": stats,
                "responses": responses,
                "entries": entries,
                "memory_scope": scope or "all",
                "show_deleted": include_deleted,
                "worker_id": worker_id,
                "current_step": current_step,
                "max_step": max_step,
                "full_prompt": full_prompt,
                "full_response": full_response,
            },
        )

    async def debug_responses(request: Request) -> JSONResponse:
        """Debug endpoint to check raw response data."""
        worker_id = request.path_params["worker_id"]
        responses = db.get_recent_responses(worker_id, limit=20)
        agents = db.get_all_agents()
        return JSONResponse({
            "worker_id": worker_id,
            "response_count": len(responses),
            "responses": responses[:5],
            "agents": [{"worker_id": a.worker_id, "episode": a.episode, "step": a.step} for a in agents],
        })

    routes = [
        Route("/", index),
        Route("/partials/agents", partials_agents),
        Route("/partials/all/{worker_id}", partials_all),
        Route("/debug/{worker_id}", debug_responses),
        Mount("/static", StaticFiles(directory=MONITOR_DIR / "static"), name="static"),
    ]

    return Starlette(routes=routes)
