"""Starlette web server for BRAID monitor."""

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
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
        current_episode = stats.episode if stats else None

        # Filter responses by current episode to avoid cross-episode confusion
        responses = db.get_recent_responses(
            worker_id, limit=10, max_step=current_step, episode=current_episode
        )

        # Memory filters from query params
        scope = request.query_params.get("scope")
        if scope not in ("persistent", "episode"):
            scope = None
        include_deleted = request.query_params.get("deleted") == "1"

        # Pass current episode for episode-scoped memory filtering
        entries = db.get_memory_entries(
            limit=50, scope=scope, include_deleted=include_deleted, episode=current_episode
        )

        # Get full prompt and response for debug panel (filter by episode)
        # Check if using SDK for incremental prompt view
        using_sdk = db.is_using_sdk(worker_id)
        sdk_history = db.get_sdk_prompt_history(worker_id, current_episode, max_step=current_step) if using_sdk and current_episode else []
        full_prompt = db.get_latest_prompt(worker_id, max_step=current_step)
        full_response = db.get_latest_full_response(
            worker_id, max_step=current_step, episode=current_episode
        )

        # Get visited positions for exploration overlay (filtered by current step when viewing history)
        visited: set[tuple[int, int]] = set()
        level_info = db.get_current_level_info(worker_id, current_episode, max_step=current_step) if current_episode else None
        if level_info is not None and current_episode is not None:
            dungeon_num, dlvl = level_info
            visited = db.get_visited_positions(worker_id, current_episode, dungeon_num, dlvl, max_step=current_step)

        # Pre-process screen with visited flags for template
        # Map row y (0-20) appears at screen row y+1 (rows 1-21)
        screen_rows: list[list[tuple[str, bool]]] = []
        if screen:
            for row_idx, row in enumerate(screen.split("\n")):
                row_data: list[tuple[str, bool]] = []
                for col_idx, char in enumerate(row):
                    is_visited = (
                        1 <= row_idx <= 21 and (col_idx, row_idx - 1) in visited
                    )
                    row_data.append((char, is_visited))
                screen_rows.append(row_data)

        # Get todos
        todos = db.get_todos(worker_id, current_episode) if current_episode else []

        # Get tool calls grouped by step (for all conversation history)
        tool_calls_by_step: dict[int, list] = {}
        if current_episode is not None:
            tool_calls_by_step = db.get_tool_calls_by_step(worker_id, current_episode)

        return templates.TemplateResponse(
            request,
            "partials/all.html",
            {
                "screen": screen,
                "screen_rows": screen_rows,
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
                "sdk_history": sdk_history,
                "using_sdk": using_sdk,
                "visited_count": len(visited),
                "tool_calls_by_step": tool_calls_by_step,
                "todos": todos,
            },
        )

    routes = [
        Route("/", index),
        Route("/partials/agents", partials_agents),
        Route("/partials/all/{worker_id}", partials_all),
        Mount("/static", StaticFiles(directory=MONITOR_DIR / "static"), name="static"),
    ]

    return Starlette(routes=routes)
