"""File and console logging setup."""

import logging
from pathlib import Path


def configure_logging(log_path: str | Path | None = None) -> None:
    """Configure concise epoch-level logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
