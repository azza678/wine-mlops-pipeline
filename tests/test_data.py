"""Unit tests for data pipeline."""
from src.data import load_wine_data, validate_data


def test_split_shape():
    X_train, X_test, y_train, y_test = load_wine_data()
    assert X_train.shape[0] == 142
    assert X_test.shape[0] == 36
    assert X_train.shape[1] == 13


def test_no_nulls_and_valid_classes():
    X_train, X_test, y_train, y_test = load_wine_data()
    assert validate_data(X_train, y_train) is True


def test_reproducibility():
    a = load_wine_data()
    b = load_wine_data()
    assert (a[0].values == b[0].values).all()
