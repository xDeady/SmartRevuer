import numpy as np
from smartrevuer.visualization import ascii_digit, color, progress_bar, section_header
from smartrevuer.config import ANSI_GREEN, ANSI_RESET


def test_ascii_digit_shape():
    pixels = np.zeros(64)
    result = ascii_digit(pixels)
    lines = result.split("\n")
    assert len(lines) == 8
    for line in lines:
        assert len(line) == 8


def test_ascii_digit_uses_all_chars():
    # Pixel values 0, 5, 10, 14 should map to ░ ▒ ▓ █
    pixels = np.array([0, 5, 10, 14] * 16)
    result = ascii_digit(pixels)
    assert "░" in result
    assert "▒" in result
    assert "▓" in result
    assert "█" in result


def test_ascii_digit_bright_pixel():
    pixels = np.full(64, 16)
    result = ascii_digit(pixels)
    assert all(c == "█" for c in result.replace("\n", ""))


def test_ascii_digit_dark_pixel():
    pixels = np.zeros(64)
    result = ascii_digit(pixels)
    assert all(c == "░" for c in result.replace("\n", ""))


def test_color_wraps_ansi():
    result = color("hello", ANSI_GREEN)
    assert result == f"{ANSI_GREEN}hello{ANSI_RESET}"


def test_progress_bar_full():
    bar = progress_bar(20, 20, width=10)
    assert "██████████" in bar
    assert "100%" in bar


def test_progress_bar_half():
    bar = progress_bar(10, 20, width=10)
    assert "█████░░░░░" in bar
    assert "50%" in bar


def test_progress_bar_zero():
    bar = progress_bar(0, 20, width=10)
    assert "░░░░░░░░░░" in bar
    assert "0%" in bar


def test_section_header_contains_step_and_title():
    header = section_header(3, "Обучение", "🧠")
    assert "3" in header
    assert "Обучение" in header
    assert "🧠" in header
    assert "ШАГ" in header
