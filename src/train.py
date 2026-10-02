"""Training pipeline with MLflow tracking for Wine classification."""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_wine_data, validate_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
SEED = 42


def get_search_grids():
    """Define hyperparameter grids for both model families."""
    rf_grid = [
        {"n_estimators": 50, "max_depth": 3, "random_state": SEED},
        {"n_estimators": 100, "max_depth": 5, "random_state": SEED},
        {"n_estimators": 200, "max_depth": None, "random_state": SEED},
    ]
    gbm_grid = [
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2, "random_state": SEED},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": SEED},
        {"n_estimators": 150, "learning_rate": 0.01, "max_depth": 3, "random_state": SEED},
    ]
    return {
        "RandomForest": (RandomForestClassifier, rf_grid),
        "GradientBoosting": (GradientBoostingClassifier, gbm_grid),
    }


def evaluate_config(model, X_train, y_train):
    """Run 5-fold stratified CV and return metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "log_loss": "neg_log_loss",
    }
    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    return {
        "train_accuracy": results["train_accuracy"].mean(),
        "train_f1_macro": results["train_f1_macro"].mean(),
        "train_log_loss": -results["train_log_loss"].mean(),
        "val_accuracy": results["test_accuracy"].mean(),
        "val_f1_macro": results["test_f1_macro"].mean(),
        "val_log_loss": -results["test_log_loss"].mean(),
    }


def train_all():
    """Main training: log all configs to MLflow, register champion."""
    X_train, X_test, y_train, y_test = load_wine_data()
    validate_data(X_train, y_train)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    grids = get_search_grids()
    best_f1 = -1
    best_run_id = None

    for family_name, (ModelClass, grid) in grids.items():
        for i, params in enumerate(grid):
            with mlflow.start_run(run_name=f"{family_name}_cfg{i}") as run:
                model = ModelClass(**params)
                metrics = evaluate_config(model, X_train, y_train)

                mlflow.log_param("model_family", family_name)
                for k, v in params.items():
                    mlflow.log_param(k, v)

                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                mlflow.set_tags({"stage": "tuning", "family": family_name})

                model.fit(X_train, y_train)
                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train.iloc[:5]

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                if metrics["val_f1_macro"] > best_f1:
                    best_f1 = metrics["val_f1_macro"]
                    best_run_id = run.info.run_id

    print(f"Best run: {best_run_id} | F1={best_f1:.4f}")
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )
    print(f"Registered {REGISTERED_MODEL_NAME} v{registered.version} as 'champion'")
    return best_run_id, registered.version


if __name__ == "__main__":
    train_all()