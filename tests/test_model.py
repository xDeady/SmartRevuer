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
