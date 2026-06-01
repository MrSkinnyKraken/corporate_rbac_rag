"""Convenience entrypoint to launch the FastAPI service.

Runs uvicorn with parameters read from :class:`~core.config.Settings`
so the api binds to the same host/port the rest of the project expects.
Use ``--reload`` for dev workflows; production deployments should use
the docker-compose ``api`` service instead.

Example::

    python -m scripts.run_api               # listens on settings.api_host:api_port
    python -m scripts.run_api --reload      # hot-reload on .py changes
"""

from __future__ import annotations

import argparse

import uvicorn

from core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Zero-Trust RAG demo API.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload on source changes (dev only).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override Settings.api_host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override Settings.api_port.",
    )
    args = parser.parse_args()
    s = get_settings()

    uvicorn.run(
        "api.main:app",
        host=args.host or s.api_host,
        port=args.port or s.api_port,
        log_level=s.api_log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
