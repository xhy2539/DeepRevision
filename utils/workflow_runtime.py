"""LangGraph durable runtime owned by FastAPI's application lifespan."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.multi_agent.supervisor import build_supervisor_graph
from utils.path_tool import get_abs_path


def checkpoint_path() -> str:
    configured = os.getenv("DEEPREVISION_CHECKPOINT_DB", "data/workflow_checkpoints.sqlite3")
    path = Path(configured if os.path.isabs(configured) else get_abs_path(configured))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@asynccontextmanager
async def durable_workflow() -> AsyncIterator[object]:
    """Create one async saver/connection per server process and close it cleanly."""
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path()) as saver:
        await saver.setup()
        yield build_supervisor_graph(checkpointer=saver)
