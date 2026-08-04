"""Small dependency-free JSON logging setup for the API and agent runtime."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_LOG_RECORD_FIELDS = set(
    logging.LogRecord(None, logging.INFO, "", 0, "", (), None).__dict__
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line so Docker and log shippers can parse it."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_context.get()
        if request_id != "-":
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Configure the application logger once, including under Uvicorn reload."""

    app_logger = logging.getLogger("library_agent")
    app_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    app_logger.propagate = False

    if any(getattr(handler, "_library_agent_json", False) for handler in app_logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler._library_agent_json = True  # type: ignore[attr-defined]
    handler.setFormatter(JsonFormatter())
    app_logger.addHandler(handler)
