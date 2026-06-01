"""Convenience entrypoint to launch the Streamlit frontend natively.

Runs Streamlit via ``streamlit.web.cli.main`` AFTER ensuring the project
root is on :data:`sys.path`. Without that prefix, the absolute imports
in ``frontend/streamlit_app.py`` (e.g. ``from frontend.api_client
import ApiClient``) fail at script-load time because
``streamlit run`` only adds the *script's directory* to ``sys.path``,
not the project root.

The dockerised frontend solves the same problem with
``ENV PYTHONPATH=/app`` in ``Dockerfile.frontend``; this script is the
host-mode equivalent for the hybrid-dev workflow.

Example::

    python -m scripts.run_frontend
    python -m scripts.run_frontend --port 8502
    API_URL=http://localhost:8000 python -m scripts.run_frontend
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit frontend.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port Streamlit binds to (default: 8501, or $STREAMLIT_PORT if set).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Address Streamlit binds to (default: 0.0.0.0).",
    )
    args = parser.parse_args()

    # Ensure the project root is on sys.path so `frontend.*` is importable
    # from the script. Done BEFORE importing streamlit so streamlit's own
    # bootstrap sees the right path layout.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Build the streamlit CLI argv equivalent to:
    #   streamlit run frontend/streamlit_app.py --server.headless true ...
    target = project_root / "frontend" / "streamlit_app.py"
    port = args.port or int(os.environ.get("STREAMLIT_PORT", 8501))
    host = args.host or os.environ.get("STREAMLIT_HOST", "0.0.0.0")

    cli_args = [
        "streamlit", "run", str(target),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    # Streamlit's CLI is built on click; calling it via sys.argv mimics the
    # command-line invocation exactly.
    sys.argv = cli_args
    from streamlit.web import cli as stcli  # noqa: PLC0415  (deferred)
    stcli.main()


if __name__ == "__main__":
    main()
