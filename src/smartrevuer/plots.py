# src/smartrevuer/plots.py
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix

from smartrevuer.config import OUTPUT_DIRS, PLOT_FILENAMES
from smartrevuer.data import DataBundle
from smartrevuer.model import ModelResult


def get_output_dir() -> Path:
    """Return first writable directory from OUTPUT_DIRS."""
    for d in OUTPUT_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.touch()
            probe.unlink()
            return d
        except (PermissionError, OSError):
            continue
    raise RuntimeError("Не удалось найти доступную папку для PNG")


def save_all_plots(
    data: DataBundle,
    result: ModelResult,
    output_dir: Path,
) -> list[Path]:
    """Save 4 PNG plots to output_dir. Returns list of Paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_digit_examples(data, output_dir / PLOT_FILENAMES[0]),
        _plot_confusion_matrix(data, result, output_dir / PLOT_FILENAMES[1]),
        _plot_accuracy_per_digit(data, result, output_dir / PLOT_FILENAMES[2]),
        _plot_network_diagram(output_dir / PLOT_FILENAMES[3]),
    ]
    return paths


def _plot_digit_examples(data: DataBundle, path: Path) -> Path:
    """2 rows × 10 cols: two different examples for each digit 0–9."""
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
    fig.suptitle("Примеры цифр из датасета", fontsize=14, fontweight="bold")

    for digit in range(10):
        indices = np.where(data.y_orig == digit)[0]
        for row in range(2):
            idx = indices[row]
            ax = axes[row, digit]
            ax.imshow(data.X_orig[idx].reshape(8, 8), cmap="gray_r", interpolation="nearest")
            if row == 0:
                ax.set_title(str(digit), fontsize=12, fontweight="bold")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrix(data: DataBundle, result: ModelResult, path: Path) -> Path:
    """Color confusion matrix heatmap with counts."""
    cm = confusion_matrix(data.y_test, result.y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Предсказание ИИ", fontsize=12)
    ax.set_ylabel("Правильный ответ", fontsize=12)
    ax.set_title("Матрица ошибок — где ИИ путается", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_accuracy_per_digit(data: DataBundle, result: ModelResult, path: Path) -> Path:
    """Horizontal bar chart: per-digit accuracy with green/yellow/red coding."""
    accs = []
    for digit in range(10):
        mask = data.y_test == digit
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((result.y_pred[mask] == digit).mean())

    colors = []
    for a in accs:
        if a >= 0.9:
            colors.append("#4CAF50")
        elif a >= 0.8:
            colors.append("#FFC107")
        else:
            colors.append("#F44336")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(10), [a * 100 for a in accs], color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Цифра {d}" for d in range(10)], fontsize=11)
    ax.set_xlabel("Точность (%)", fontsize=12)
    ax.set_title("Точность ИИ по каждой цифре", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}",
            va="center",
            fontsize=10,
        )

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="≥ 90% (отлично)"),
        mpatches.Patch(color="#FFC107", label="≥ 80% (хорошо)"),
        mpatches.Patch(color="#F44336", label="< 80% (учимся)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_network_diagram(path: Path) -> Path:
    """Architecture diagram: layers as colored boxes with arrows."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    layers = [
        ("Вход\n64 пикселя\n(8×8)", 1.5, 2.5, "#4CAF50", "white"),
        ("ReLU\n128 нейронов\nСкрытый 1", 4.5, 2.5, "#2196F3", "white"),
        ("ReLU\n64 нейрона\nСкрытый 2", 7.5, 2.5, "#2196F3", "white"),
        ("Softmax\n10 классов\n(цифры 0–9)", 10.5, 2.5, "#FF9800", "white"),
    ]

    box_positions = []
    for label, x, y, fc, tc in layers:
        bbox = FancyBboxPatch(
            (x - 1.1, y - 1.0), 2.2, 2.0,
            boxstyle="round,pad=0.15",
            facecolor=fc,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, color=tc, fontweight="bold", zorder=3)
        box_positions.append((x, y))

    for i in range(len(box_positions) - 1):
        x1, y1 = box_positions[i][0] + 1.1, box_positions[i][1]
        x2, y2 = box_positions[i + 1][0] - 1.1, box_positions[i + 1][1]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=2.5),
            zorder=1,
        )

    ax.text(6.5, 4.5, "Нейронная сеть SmartRevuer",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(6.5, 0.3,
            "Каждая стрелка — это тысячи весов, которые ИИ настроил во время обучения",
            ha="center", va="center", fontsize=9, color="#666666", style="italic")

    fig.patch.set_facecolor("#F8F9FA")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
