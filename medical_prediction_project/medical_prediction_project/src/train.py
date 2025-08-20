import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, classification_report,
                             confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)
import joblib

from utils import save_json

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "breast_cancer.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    # Export transparency dataset
    df = data.frame.copy()
    df.to_csv(DATA_RAW, index=False)
    return X, y, data

def get_models_and_params():
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear"))
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
        ]),
        "gb": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
    }
    params = {
        "logreg": {"clf__C": [0.1, 1.0, 3.0, 10.0]},
        "rf": {"clf__max_depth": [None, 4, 8, 16], "clf__min_samples_leaf": [1, 2, 4]},
        "gb": {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]},
    }
    return models, params

def evaluate_and_plot(model, X_test, y_test, prefix="best"):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    REPORTS.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    roc_path = REPORTS / f"{prefix}_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # PR Curve
    plt.figure()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    pr_path = REPORTS / f"{prefix}_pr.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix heatmap-like plot
    plt.figure()
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=["Malignant(0)", "Benign(1)"],
                yticklabels=["Malignant(0)", "Benign(1)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = REPORTS / f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    metrics = {
        "roc_auc": float(roc_auc),
        "avg_precision": float(pr_auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "plots": {
            "roc": str(roc_path.name),
            "pr": str(pr_path.name),
            "confusion_matrix": str(cm_path.name)
        }
    }
    return metrics

def main():
    X, y, data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models, params = get_models_and_params()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model = None
    best_score = -np.inf
    all_results = {}

    for name, pipe in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(
            pipe, params[name], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        val_score = grid.best_score_
        print(f"{name} best CV ROC-AUC: {val_score:.4f} => {grid.best_params_}")

        test_metrics = evaluate_and_plot(grid.best_estimator_, X_test, y_test, prefix=name)
        all_results[name] = {
            "cv_best_score_roc_auc": float(val_score),
            "best_params": grid.best_params_,
            "test_metrics": test_metrics
        }

        if val_score > best_score:
            best_score = val_score
            best_model = grid.best_estimator_

    # Save best model
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path} (CV ROC-AUC={best_score:.4f})")

    # Save metrics
    save_json(all_results, REPORTS / "metrics.json")

if __name__ == "__main__":
    main()
