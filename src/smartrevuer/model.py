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
