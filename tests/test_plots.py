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
