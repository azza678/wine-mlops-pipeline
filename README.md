# Wine MLOps Pipeline

End-to-end MLOps pipeline for multi-class Wine Cultivar Classification using scikit-learn, MLflow, GitHub Actions, and Makefile automation.

## Project Overview

- **Dataset:** sklearn.datasets.load_wine (178 samples, 13 features, 3 classes)
- **Models:** RandomForestClassifier & GradientBoostingClassifier
- **Tracking:** MLflow (experiments, signatures, model registry)
- **CI/CD:** GitHub Actions with MLOps Quality Gate
- **Reproducibility:** Fixed random seed (42)

## Setup

    make install

## Lint

    make lint

## Test

    make test

## Train

    make train

## Evaluate

    python src/evaluate.py

## MLflow UI

    mlflow ui

Then open http://127.0.0.1:5000 in your browser.

## MLOps Quality Gate

The CI pipeline enforces:

- Validation Macro F1-score >= 0.88
- Batch inference latency <= 30 ms
- Output schema integrity (classes 0, 1, 2 only)

## MLflow Tracking

- Experiment name: `Wine-Cultivar-Classification`
- Registered model: `WineClassifier`
- Champion alias: `champion`
- Minimum 6 runs logged with params, metrics, and signatures

## Feature: MLflow Tracking Documentation

This branch adds detailed documentation for MLflow experiment tracking.

## Merge Resolution

MERGED-VERSION: Both RandomForest and GradientBoosting tuning configs are documented. 
BRANCH-VERSION: RandomForest tuning config 
BRANCH-VERSION: RandomForest tuning config 
BRANCH-VERSION: RandomForest tuning config
