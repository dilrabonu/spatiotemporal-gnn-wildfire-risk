"""Structured logging with rich console output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console()


def get_logger(
    name: str = "wildfire_gnn",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Create and return a logger with rich console output.

    Parameters
    ----------
    name : str
        Logger name. Use __name__ in each module.
    level : int
        Logging level. Default INFO.
    log_file : str | Path, optional
        If provided, also write logs to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Rich console handler
    console_handler = RichHandler(
        console=_console,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Optional file handler (plain text, no rich markup)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def section(title: str) -> None:
    """Print a rich section header to the console."""
    _console.rule(f"[bold]{title}[/bold]")


def success(message: str) -> None:
    """Print a green success message."""
    _console.print(f"[bold green]✓[/bold green]  {message}")


def warn(message: str) -> None:
    """Print a yellow warning message."""
    _console.print(f"[bold yellow]![/bold yellow]  {message}")


def error(message: str) -> None:
    """Print a red error message."""
    _console.print(f"[bold red]✗[/bold red]  {message}", file=sys.stderr)