"""Deployment entrypoint for the OpenEnv API server."""

from __future__ import annotations

import uvicorn

from openenv_api import app


def main() -> None:
    """Run the HTTP server used by local/dev deployment commands."""
    uvicorn.run("openenv_api:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
