"""
Smart Support Env – FastAPI Server (app.py)
=========================================
Entry point for the OpenEnv HTTP + WebSocket server.
Start with:
    uv run server          (via pyproject.toml script)
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os
import sys

# ─── Ensure project root on path ─────────────────────────────────────────────
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from openenv.core.env_server.http_server import create_app
import client
from server.smart_support_env_environment import SmartSupportEnvironment, TASK_NAMES

# ─── Create the standard OpenEnv FastAPI app ─────────────────────────────────
app = create_app(
    SmartSupportEnvironment,
    client.SmartSupportAction,
    client.SmartSupportObservation,
    env_name="smart_support_env",
    max_concurrent_envs=5,
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def get_tasks():
    """Return all supported task names (required by judge checklist)."""
    return {"tasks": TASK_NAMES}


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()