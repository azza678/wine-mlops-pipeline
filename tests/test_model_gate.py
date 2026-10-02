"""MLOps Quality Gate: F1, latency, and output schema checks."""
import time
import mlflow
import mlflow.sklearn
import numpy as np

from src.data import load_wine_data


REGISTERED_MODEL_NAME = "WineClassifier"
F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0


def _load_champion():
    mlflow.set_tracking_uri("file:./mlruns")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    return mlflow.sklearn.load_model(model_uri)


def test_validation_f1_gate():
    from sklearn.metrics import f1_score
    X_train, X_test, y_train, y_test = load_wine_data()
    model = _load_champion()
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    assert f1 >= F1_THRESHOLD, f"F1 {f1:.4f} < {F1_THRESHOLD}"


def test_inference_latency_gate():
    X_train, X_test, _, _ = load_wine_data()
    model = _load_champion()
    batch = X_test.iloc[:10]
    start = time.perf_counter()
    _ = model.predict(batch)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= LATENCY_THRESHOLD_MS, f"Latency {elapsed_ms:.2f}ms > 30ms"


def test_output_schema_gate():
    X_train, X_test, _, _ = load_wine_data()
    model = _load_champion()
    preds = model.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1, 2})
