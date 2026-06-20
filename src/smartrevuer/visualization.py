import numpy as np

from smartrevuer.config import (
    ANSI_BOLD,
    ANSI_RESET,
    ASCII_CHARS,
    ASCII_THRESHOLDS,
)


def ascii_digit(pixels: np.ndarray) -> str:
    """Render a (64,) pixel array (values 0–16) as 8×8 ASCII art."""
    grid = pixels.reshape(8, 8)
    lines = []
    for row in grid:
        line = ""
        for val in row:
            if val < ASCII_THRESHOLDS[0]:
                line += ASCII_CHARS[0]
            elif val < ASCII_THRESHOLDS[1]:
                line += ASCII_CHARS[1]
            elif val < ASCII_THRESHOLDS[2]:
                line += ASCII_CHARS[2]
            else:
                line += ASCII_CHARS[3]
        lines.append(line)
    return "\n".join(lines)


def color(text: str, code: str) -> str:
    """Wrap text in an ANSI color code."""
    return f"{code}{text}{ANSI_RESET}"


def progress_bar(value: float, max_value: float, width: int = 20) -> str:
    """Return a █░ progress bar string with percentage."""
    filled = int(width * value / max_value) if max_value > 0 else 0
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * value / max_value) if max_value > 0 else 0
    return f"{bar} {pct}%"


def section_header(step: int, title: str, icon: str) -> str:
    """Return a bold section header line."""
    line = f"─── {icon} ШАГ {step}: {title} ───"
    return color(line, ANSI_BOLD)
