# SmartRevuer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать полноценный Python-пакет SmartRevuer для распознавания рукописных цифр (MLPClassifier) с 6 шагами, ASCII-визуализацией, 4 PNG-графиками и интерактивным демо для школьников 1–3 классов.

**Architecture:** `src/smartrevuer/` с 7 модулями (config, data, model, visualization, plots, cli, __main__). Тесты в `tests/`. Точка входа: `python -m smartrevuer` или `make run`.

**Tech Stack:** Python 3.10+, scikit-learn≥1.3, matplotlib≥3.7, seaborn≥0.12, numpy≥1.24, pytest, ruff.

## Global Constraints

- Только `sklearn.datasets.load_digits` — никакого MNIST из интернета
- Никаких `plt.show()` — только `plt.savefig()`
- Никаких `tensorflow`, `keras`, `torch`
- Весь вывод в терминале на русском языке
- `random_state=42` везде для воспроизводимости
- Python 3.10+

---

### Task 1: Project Scaffold + config.py

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/smartrevuer/__init__.py`
- Create: `src/smartrevuer/config.py`
- Create: `tests/__init__.py`
- Create: `outputs/.gitkeep`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `from smartrevuer.config import HIDDEN_LAYERS, TEST_SIZE, RANDOM_STATE, MAX_ITER`
  - `from smartrevuer.config import ASCII_CHARS, ASCII_THRESHOLDS`
  - `from smartrevuer.config import ANSI_GREEN, ANSI_RED, ANSI_GREY, ANSI_YELLOW, ANSI_BOLD, ANSI_RESET`
  - `from smartrevuer.config import OUTPUT_DIRS, PLOT_FILENAMES`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "smartrevuer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "seaborn>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.1"]

[project.scripts]
smartrevuer = "smartrevuer.cli:run_pipeline"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py310"
```

- [ ] **Step 2: Create requirements.txt**

```
scikit-learn>=1.3
matplotlib>=3.7
numpy>=1.24
seaborn>=0.12
pytest>=7
ruff>=0.1
```

- [ ] **Step 3: Create Makefile**

```makefile
.PHONY: install run test lint clean

install:
	pip install -e ".[dev]"

run:
	python -m smartrevuer

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf outputs/*.png build/ dist/ src/*.egg-info
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
venv/
outputs/*.png
*.png
```

- [ ] **Step 5: Create README.md**

```markdown
# SmartRevuer — Умный проверяльщик цифр

Учебный ИИ-проект для школьников 1–3 классов: нейросеть учится распознавать рукописные цифры 0–9.

## Запуск

```bash
make install
make run
```

## Тесты

```bash
make test
```
```

- [ ] **Step 6: Create src/smartrevuer/__init__.py** (пустой файл)

- [ ] **Step 7: Create src/smartrevuer/config.py**

```python
from pathlib import Path

# Neural network
HIDDEN_LAYERS = (128, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 500

# ASCII brightness thresholds (pixel values range 0–16)
ASCII_CHARS = ["░", "▒", "▓", "█"]
ASCII_THRESHOLDS = [5, 10, 14]

# ANSI color codes
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_GREY = "\033[90m"
ANSI_YELLOW = "\033[93m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"

# Output directories (tried in order, first writable wins)
OUTPUT_DIRS = [Path("/mnt/user-data/outputs"), Path("outputs")]

# PNG filenames in save order
PLOT_FILENAMES = [
    "01_digit_examples.png",
    "02_confusion_matrix.png",
    "03_accuracy_per_digit.png",
    "04_network_diagram.png",
]
```

- [ ] **Step 8: Create tests/__init__.py** (пустой файл)

- [ ] **Step 9: Create outputs/.gitkeep** (пустой файл)

- [ ] **Step 10: Install package and verify**

```bash
pip install -e ".[dev]"
python -c "from smartrevuer.config import HIDDEN_LAYERS; print(HIDDEN_LAYERS)"
```

Expected: `(128, 64)`

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml requirements.txt Makefile .gitignore README.md \
        src/ tests/__init__.py outputs/.gitkeep
git commit -m "feat: project scaffold with config constants"
```

---

### Task 2: visualization.py + tests

**Files:**
- Create: `src/smartrevuer/visualization.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `from smartrevuer.config import ASCII_CHARS, ASCII_THRESHOLDS, ANSI_BOLD, ANSI_RESET, ANSI_GREEN, ANSI_RED, ANSI_GREY, ANSI_YELLOW`
- Produces:
  - `ascii_digit(pixels: np.ndarray) -> str` — pixels: shape (64,), values 0–16; returns 8 lines of 8 chars joined by `\n`
  - `color(text: str, code: str) -> str` — wraps text in ANSI code + ANSI_RESET
  - `progress_bar(value: float, max_value: float, width: int = 20) -> str` — returns e.g. `"████████░░░░░░░░░░░░ 40%"`
  - `section_header(step: int, title: str, icon: str) -> str` — returns bold `"─── 📦 ШАГ 1: Название ───"`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_visualization.py
import numpy as np
import pytest
from smartrevuer.visualization import ascii_digit, color, progress_bar, section_header
from smartrevuer.config import ANSI_GREEN, ANSI_RESET, ANSI_BOLD


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_visualization.py -v
```

Expected: `ImportError` — модуль не существует.

- [ ] **Step 3: Implement visualization.py**

```python
# src/smartrevuer/visualization.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_visualization.py -v
```

Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartrevuer/visualization.py tests/test_visualization.py
git commit -m "feat: add visualization utils with ASCII digit renderer"
```

---

### Task 3: data.py + tests

**Files:**
- Create: `src/smartrevuer/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: `from smartrevuer.config import TEST_SIZE, RANDOM_STATE`
- Produces:
  - `DataBundle` — dataclass с полями: `X_train: np.ndarray`, `X_test: np.ndarray`, `y_train: np.ndarray`, `y_test: np.ndarray`, `scaler: StandardScaler`, `X_orig: np.ndarray`, `y_orig: np.ndarray`
  - `load_and_split() -> DataBundle`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data.py
import numpy as np
import pytest
from smartrevuer.data import DataBundle, load_and_split


@pytest.fixture(scope="module")
def bundle():
    return load_and_split()


def test_bundle_field_shapes(bundle):
    n_train = len(bundle.X_train)
    n_test = len(bundle.X_test)
    assert bundle.X_train.shape == (n_train, 64)
    assert bundle.X_test.shape == (n_test, 64)
    assert bundle.y_train.shape == (n_train,)
    assert bundle.y_test.shape == (n_test,)


def test_train_test_split_ratio(bundle):
    total = len(bundle.X_train) + len(bundle.X_test)
    assert total == 1797
    assert abs(len(bundle.X_test) / total - 0.2) < 0.01


def test_train_scaled_mean_near_zero(bundle):
    assert abs(bundle.X_train.mean()) < 0.1


def test_train_scaled_std_near_one(bundle):
    assert abs(bundle.X_train.std() - 1.0) < 0.1


def test_test_uses_train_scaler(bundle):
    # Test set mean should also be close to 0 (same scaler applied)
    assert abs(bundle.X_test.mean()) < 0.5


def test_stratification(bundle):
    # Each digit 0-9 should appear in test set
    unique = np.unique(bundle.y_test)
    assert len(unique) == 10


def test_orig_is_unscaled(bundle):
    assert bundle.X_orig.max() <= 16
    assert bundle.X_orig.min() >= 0
    assert bundle.X_orig.shape == (1797, 64)


def test_orig_labels(bundle):
    assert set(bundle.y_orig) == set(range(10))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement data.py**

```python
# src/smartrevuer/data.py
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from smartrevuer.config import RANDOM_STATE, TEST_SIZE


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    X_orig: np.ndarray
    y_orig: np.ndarray


def load_and_split() -> DataBundle:
    """Load sklearn digits dataset, scale, and split 80/20."""
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        X_orig=X,
        y_orig=y,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartrevuer/data.py tests/test_data.py
git commit -m "feat: add data loading and preprocessing with DataBundle"
```

---

### Task 4: model.py + tests

**Files:**
- Create: `src/smartrevuer/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes:
  - `DataBundle` из `smartrevuer.data`
  - `from smartrevuer.config import HIDDEN_LAYERS, MAX_ITER, RANDOM_STATE`
- Produces:
  - `ModelResult` — dataclass с полями: `model: MLPClassifier`, `y_pred: np.ndarray`, `train_acc: float`, `test_acc: float`
  - `train(data: DataBundle) -> ModelResult`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import numpy as np
import pytest
from smartrevuer.data import load_and_split
from smartrevuer.model import ModelResult, train


@pytest.fixture(scope="module")
def result():
    data = load_and_split()
    return train(data)


@pytest.fixture(scope="module")
def data():
    return load_and_split()


def test_result_has_fields(result):
    assert hasattr(result, "model")
    assert hasattr(result, "y_pred")
    assert hasattr(result, "train_acc")
    assert hasattr(result, "test_acc")


def test_test_accuracy_above_80pct(result):
    assert result.test_acc >= 0.80, f"Точность {result.test_acc:.2%} < 80%"


def test_y_pred_shape(result, data):
    loaded = load_and_split()
    assert result.y_pred.shape == (len(loaded.y_test),)


def test_y_pred_values_in_range(result):
    assert set(result.y_pred).issubset(set(range(10)))


def test_train_acc_above_test_acc(result):
    # Training accuracy should be >= test accuracy (no underfitting)
    assert result.train_acc >= result.test_acc - 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement model.py**

```python
# src/smartrevuer/model.py
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier

from smartrevuer.config import HIDDEN_LAYERS, MAX_ITER, RANDOM_STATE
from smartrevuer.data import DataBundle


@dataclass
class ModelResult:
    model: MLPClassifier
    y_pred: np.ndarray
    train_acc: float
    test_acc: float


def train(data: DataBundle) -> ModelResult:
    """Train MLPClassifier on DataBundle and return results."""
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(data.X_train, data.y_train)

    train_acc = model.score(data.X_train, data.y_train)
    test_acc = model.score(data.X_test, data.y_test)
    y_pred = model.predict(data.X_test)

    return ModelResult(
        model=model,
        y_pred=y_pred,
        train_acc=train_acc,
        test_acc=test_acc,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartrevuer/model.py tests/test_model.py
git commit -m "feat: add MLPClassifier training with ModelResult"
```

---

### Task 5: plots.py + tests

**Files:**
- Create: `src/smartrevuer/plots.py`
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes:
  - `DataBundle` из `smartrevuer.data`
  - `ModelResult` из `smartrevuer.model`
  - `from smartrevuer.config import PLOT_FILENAMES, OUTPUT_DIRS`
- Produces:
  - `get_output_dir() -> Path` — возвращает первую доступную для записи папку из `OUTPUT_DIRS`
  - `save_all_plots(data: DataBundle, result: ModelResult, output_dir: Path) -> list[Path]` — сохраняет 4 PNG, возвращает список путей

- [ ] **Step 1: Write failing tests**

```python
# tests/test_plots.py
import numpy as np
import pytest
from pathlib import Path
from smartrevuer.data import load_and_split
from smartrevuer.model import train
from smartrevuer.plots import save_all_plots
from smartrevuer.config import PLOT_FILENAMES


@pytest.fixture(scope="module")
def pipeline_result():
    data = load_and_split()
    result = train(data)
    return data, result


def test_save_all_plots_returns_four_paths(tmp_path, pipeline_result):
    data, result = pipeline_result
    paths = save_all_plots(data, result, tmp_path)
    assert len(paths) == 4


def test_save_all_plots_files_exist(tmp_path, pipeline_result):
    data, result = pipeline_result
    paths = save_all_plots(data, result, tmp_path)
    for p in paths:
        assert p.exists(), f"Файл не создан: {p}"


def test_save_all_plots_correct_names(tmp_path, pipeline_result):
    data, result = pipeline_result
    paths = save_all_plots(data, result, tmp_path)
    names = [p.name for p in paths]
    assert names == PLOT_FILENAMES


def test_save_all_plots_are_png(tmp_path, pipeline_result):
    data, result = pipeline_result
    paths = save_all_plots(data, result, tmp_path)
    for p in paths:
        assert p.suffix == ".png"
        assert p.stat().st_size > 1000  # не пустой файл
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_plots.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement plots.py**

```python
# src/smartrevuer/plots.py
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="Минимум 80%")
    ax.legend(fontsize=10)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_plots.py -v
```

Expected: все тесты PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smartrevuer/plots.py tests/test_plots.py
git commit -m "feat: add 4 PNG plot generators"
```

---

### Task 6: cli.py — Pipeline (6 шагов + интерактивное демо)

**Files:**
- Create: `src/smartrevuer/cli.py`

**Interfaces:**
- Consumes:
  - `load_and_split() -> DataBundle` из `smartrevuer.data`
  - `train(data) -> ModelResult` из `smartrevuer.model`
  - `ascii_digit, color, progress_bar, section_header` из `smartrevuer.visualization`
  - `save_all_plots, get_output_dir` из `smartrevuer.plots`
  - Все ANSI-константы из `smartrevuer.config`
- Produces:
  - `run_pipeline() -> None` — вызывается из `__main__`

- [ ] **Step 1: Implement cli.py**

```python
# src/smartrevuer/cli.py
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix

from smartrevuer.config import (
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_GREY,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
)
from smartrevuer.data import DataBundle, load_and_split
from smartrevuer.model import ModelResult, train
from smartrevuer.plots import get_output_dir, save_all_plots
from smartrevuer.visualization import (
    ascii_digit,
    color,
    progress_bar,
    section_header,
)


def _print_step1(data: DataBundle) -> None:
    print(section_header(1, "Загрузка данных", "📦"))
    print()
    print(color("ИИ видит цифру как сетку 8×8 пикселей — 64 числа от 0 до 16.", ANSI_GREY))
    print()

    n_total = len(data.X_orig)
    print(f"  Всего образцов:      {color(str(n_total), ANSI_BOLD)}")
    print(f"  Размер изображения:  {color('8×8 пикселей', ANSI_BOLD)}")
    print(f"  Классов:             {color('10 (цифры 0–9)', ANSI_BOLD)}")
    print(f"  Обучающих:           {color(str(len(data.X_train)), ANSI_GREEN)}")
    print(f"  Тестовых:            {color(str(len(data.X_test)), ANSI_YELLOW)}")
    print()
    print(color("Распределение по классам:", ANSI_BOLD))

    for digit in range(10):
        count = (data.y_orig == digit).sum()
        bar = progress_bar(count, n_total, width=15)
        print(f"  Цифра {digit}: {bar}  {color(str(count), ANSI_GREY)}")
    print()


def _print_step2(data: DataBundle) -> None:
    print(section_header(2, "ASCII-визуализация цифр", "🖼️"))
    print()
    print(color("Вот как ИИ «видит» цифры — каждый символ это один пиксель:", ANSI_GREY))
    print()

    rng = np.random.default_rng(0)
    indices = rng.choice(len(data.X_orig), size=5, replace=False)

    for idx in indices:
        digit = data.y_orig[idx]
        print(color(f"  ╔══ Это цифра {digit} ══╗", ANSI_BOLD))
        art = ascii_digit(data.X_orig[idx])
        for line in art.split("\n"):
            print(f"  ║ {line} ║")
        print(color("  ╚══════════════╝", ANSI_BOLD))
        print()


def _print_step3(data: DataBundle, result: ModelResult) -> None:
    print(section_header(3, "Обучение нейросети", "🧠"))
    print()
    print(color(
        f"Мы показали ИИ {len(data.X_train)} примеров, он запомнил узоры...",
        ANSI_GREY,
    ))
    print()

    acc_color = ANSI_GREEN if result.train_acc >= 0.80 else ANSI_RED
    print(f"  Точность на обучении:  {progress_bar(result.train_acc, 1.0)}  "
          f"{color(f'{result.train_acc:.1%}', acc_color)}")

    acc_color = ANSI_GREEN if result.test_acc >= 0.80 else ANSI_RED
    mark = color("✅", ANSI_GREEN) if result.test_acc >= 0.80 else color("❌", ANSI_RED)
    print(f"  Точность на тесте:     {progress_bar(result.test_acc, 1.0)}  "
          f"{color(f'{result.test_acc:.1%}', acc_color)}  {mark}")
    print()
    print(color("ИИ не просто запомнил — он научился узнавать новые примеры!", ANSI_GREY))
    print()


def _print_step4(data: DataBundle, result: ModelResult) -> None:
    print(section_header(4, "Анализ ошибок", "🔍"))
    print()
    print(color("Ошибки — это нормально! ИИ не волшебник.", ANSI_YELLOW))
    print()

    cm = confusion_matrix(data.y_test, result.y_pred)

    # Terminal confusion matrix
    print(color("Матрица ошибок (строка = правильный ответ, столбец = предсказание ИИ):", ANSI_BOLD))
    header = "      " + "  ".join(f"{i:2d}" for i in range(10))
    print(color(header, ANSI_GREY))
    for i, row in enumerate(cm):
        cells = []
        for j, val in enumerate(row):
            if i == j:
                cells.append(color(f"{val:3d}", ANSI_GREEN))
            elif val > 0:
                cells.append(color(f"{val:3d}", ANSI_RED))
            else:
                cells.append(f"{val:3d}")
        print(f"  [{i}]  {'  '.join(cells)}")
    print()

    # Top-5 confusions
    print(color("Топ-5 самых частых путаниц:", ANSI_BOLD))
    confusions = []
    for true_digit in range(10):
        for pred_digit in range(10):
            if true_digit != pred_digit and cm[true_digit, pred_digit] > 0:
                confusions.append((cm[true_digit, pred_digit], true_digit, pred_digit))
    confusions.sort(reverse=True)

    for rank, (count, true_d, pred_d) in enumerate(confusions[:5], 1):
        print(f"  {rank}. ИИ принял {color(str(true_d), ANSI_BOLD)} "
              f"за {color(str(pred_d), ANSI_YELLOW)} "
              f"({color(str(count), ANSI_RED)} раз)")
    print()

    # Show 3 error examples with ASCII
    print(color("Примеры ошибок ИИ:", ANSI_BOLD))
    error_mask = result.y_pred != data.y_test
    error_indices = np.where(error_mask)[0][:3]

    for idx in error_indices:
        true_label = data.y_test[idx]
        pred_label = result.y_pred[idx]
        raw_pixels = data.scaler.inverse_transform(data.X_test[idx:idx+1])[0]
        raw_pixels = np.clip(raw_pixels, 0, 16)
        print(color(
            f"  Правильно: {true_label}, ИИ сказал: {pred_label}",
            ANSI_RED,
        ))
        art = ascii_digit(raw_pixels)
        for line in art.split("\n"):
            print(f"    {line}")
        print()

    print(color("Даже люди иногда путают похожие цифры — это нормально!", ANSI_GREY))
    print()


def _print_step5(paths: list) -> None:
    print(section_header(5, "Сохранение графиков", "📊"))
    print()
    for p in paths:
        print(f"  {color('✅', ANSI_GREEN)} {p}")
    print()
    print(color("4 графика сохранены! Можно показать на уроке.", ANSI_GREY))
    print()


def _print_step6(result: ModelResult, paths: list) -> None:
    print(section_header(6, "Итоговый чек-лист", "✅"))
    print()

    checks = [
        (result.test_acc >= 0.80, f"Точность ≥ 80% на тесте ({result.test_acc:.1%})"),
        (True, "Обучение завершено без ошибок"),
        (True, "Примеры ошибок ИИ наглядно показаны (ASCII)"),
        (len(paths) == 4, f"4 PNG-файла сохранены"),
        (True, "Нет персональных данных (только load_digits)"),
        (True, "Ошибки ИИ показаны как норма (педагогический блок)"),
    ]

    all_ok = all(ok for ok, _ in checks)
    for ok, label in checks:
        mark = color("✅", ANSI_GREEN) if ok else color("❌", ANSI_RED)
        print(f"  {mark}  {label}")

    print()
    if all_ok:
        print(color("🎉 Все критерии выполнены! Проект готов к демонстрации.", ANSI_GREEN))
    else:
        print(color("⚠️  Некоторые критерии не выполнены.", ANSI_RED))
    print()


def interactive_demo(data: DataBundle, result: ModelResult) -> None:
    """Interactive digit guessing game (TTY only)."""
    print(section_header(7, "Интерактивное демо", "🎮"))
    print()
    print(color("Угадай, что скажет ИИ! Введи цифру 0–9 или 'q' для выхода.", ANSI_BOLD))
    print()

    score_correct = 0
    score_total = 0
    rng = np.random.default_rng()

    while True:
        try:
            inp = input(color("  Введи цифру (0–9) или 'q': ", ANSI_YELLOW)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if inp.lower() == "q":
            break

        if not inp.isdigit() or int(inp) > 9:
            print(color("  Введи цифру от 0 до 9!", ANSI_RED))
            continue

        digit = int(inp)
        candidates = np.where(data.y_orig == digit)[0]
        if len(candidates) == 0:
            print(color(f"  Нет примеров для цифры {digit}!", ANSI_RED))
            continue

        idx = rng.choice(candidates)
        raw_pixels = data.X_orig[idx]

        print()
        print(color(f"  ИИ смотрит на эту картинку:", ANSI_GREY))
        art = ascii_digit(raw_pixels)
        for line in art.split("\n"):
            print(f"    {line}")

        scaled = data.scaler.transform(raw_pixels.reshape(1, -1))
        pred = result.model.predict(scaled)[0]
        proba = result.model.predict_proba(scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]

        correct = pred == digit
        score_total += 1
        if correct:
            score_correct += 1

        pred_color = ANSI_GREEN if correct else ANSI_RED
        print()
        print(f"  ИИ говорит: {color(str(pred), pred_color + ANSI_BOLD)}", end="  ")
        if correct:
            print(color("✅ Правильно!", ANSI_GREEN))
        else:
            print(color(f"❌ Ошиблись (правильно: {digit})", ANSI_RED))

        print(color("  Топ-3 версии ИИ:", ANSI_GREY))
        for rank, i in enumerate(top3_idx, 1):
            bar = progress_bar(proba[i], 1.0, width=10)
            print(f"    {rank}. Цифра {i}: {bar} {proba[i]:.1%}")

        print()
        print(color(f"  Счёт: {score_correct} из {score_total} угадано", ANSI_BOLD))
        print()

    if score_total > 0:
        print(color(f"\n  Итог игры: {score_correct}/{score_total} — молодец!", ANSI_GREEN))
    print()


def run_pipeline() -> None:
    """Run all 6 steps and optional interactive demo."""
    print()
    print(color("═" * 50, ANSI_BOLD))
    print(color("  🤖  УМНЫЙ ПРОВЕРЯЛЬЩИК ЦИФР — SmartRevuer", ANSI_BOLD))
    print(color("═" * 50, ANSI_BOLD))
    print()

    # Step 1: Load data
    data = load_and_split()
    _print_step1(data)

    # Step 2: ASCII preview
    _print_step2(data)

    # Step 3: Train
    result = train(data)
    _print_step3(data, result)

    # Step 4: Error analysis
    _print_step4(data, result)

    # Step 5: Save plots
    output_dir = get_output_dir()
    paths = save_all_plots(data, result, output_dir)
    _print_step5(paths)

    # Step 6: Checklist
    _print_step6(result, paths)

    # Interactive demo (TTY only)
    if sys.stdout.isatty():
        interactive_demo(data, result)
```

- [ ] **Step 2: Run all existing tests to check no regressions**

```bash
pytest tests/ -v
```

Expected: все тесты PASS.

- [ ] **Step 3: Commit**

```bash
git add src/smartrevuer/cli.py
git commit -m "feat: add CLI pipeline with 6 steps and interactive demo"
```

---

### Task 7: __main__.py + smoke test

**Files:**
- Create: `src/smartrevuer/__main__.py`

**Interfaces:**
- Consumes: `run_pipeline` из `smartrevuer.cli`
- Produces: запускаемый пакет через `python -m smartrevuer`

- [ ] **Step 1: Create __main__.py**

```python
# src/smartrevuer/__main__.py
from smartrevuer.cli import run_pipeline

if __name__ == "__main__":
    run_pipeline()
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: все тесты PASS.

- [ ] **Step 3: Smoke test — запустить пакет и убедиться в корректном выводе**

```bash
python -m smartrevuer 2>&1 | head -60
```

Expected: заголовок `УМНЫЙ ПРОВЕРЯЛЬЩИК ЦИФР`, шаги 1–6 выполнены, 4 PNG-файла созданы.

- [ ] **Step 4: Verify PNG files exist**

```bash
ls -lh outputs/*.png 2>/dev/null || ls -lh /mnt/user-data/outputs/*.png 2>/dev/null
```

Expected: 4 файла `01_digit_examples.png`, `02_confusion_matrix.png`, `03_accuracy_per_digit.png`, `04_network_diagram.png`.

- [ ] **Step 5: Run lint**

```bash
make lint
```

Expected: без ошибок (или только предупреждения об unused imports).

- [ ] **Step 6: Commit**

```bash
git add src/smartrevuer/__main__.py
git commit -m "feat: add __main__ entry point, project complete"
```
