import numpy as np
import pytest
from smartrevuer.data import load_and_split


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
