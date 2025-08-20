from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def main():
    # Load model
    model = joblib.load(MODELS / "best_model.joblib")

    # Load data
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Average Precision (PR AUC):", average_precision_score(y_test, y_proba))

if __name__ == "__main__":
    main()
