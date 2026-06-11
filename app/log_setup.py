"""UTF-8 safe logging for CLI runners (Windows cp949 consoles)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def ensure_utf8_stdio() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass


def configure_runner_logging(
    logger: logging.Logger,
    output_dir: str | Path,
    log_filename: str,
) -> None:
    ensure_utf8_stdio()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    logfile = logging.FileHandler(out / log_filename, encoding="utf-8")
    logfile.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(logfile)
