"""Data loading and validation for Wine dataset."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


def load_wine_data(random_state: int = 42):
    """Load Wine dataset and perform stratified 80/20 split."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def validate_data(X: pd.DataFrame, y: pd.Series) -> bool:
    """Validate data: no nulls, 13 features."""
    assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"
    assert X.isnull().sum().sum() == 0, "Null values found in features"
    assert y.isnull().sum() == 0, "Null values found in target"
    assert set(y.unique()).issubset({0, 1, 2}), "Invalid class labels"
    return True


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_wine_data()
    validate_data(X_train, y_train)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
