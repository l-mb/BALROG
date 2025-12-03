"""Entry point for BRAID monitor: python -m balrog.agents.braid.monitor"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="BRAID Agent Monitor")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("memory/braid.db"),
        help="Path to braid.db SQLite database",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}")
        print("Run a BRAID agent first to create the database.")
        raise SystemExit(1)

    # Import here to defer dependency check
    try:
        import uvicorn
    except ImportError:
        print("Error: Missing dependencies. Install with:")
        print("  pip install 'balrog[monitor]'")
        print("  # or: pip install starlette uvicorn jinja2")
        raise SystemExit(1)

    from .server import create_app

    app = create_app(args.db)
    print(f"BRAID Monitor: http://{args.host}:{args.port}")
    print(f"Database: {args.db}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
