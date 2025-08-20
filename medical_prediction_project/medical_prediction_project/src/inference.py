import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def load_feature_names():
    data = load_breast_cancer(as_frame=True)
    return list(data.feature_names)

def predict_from_dict(model, features: dict):
    # Arrange features in the same order the model expects
    columns = load_feature_names()
    x = np.array([features[c] for c in columns], dtype=float).reshape(1, -1)
    proba = model.predict_proba(x)[:, 1][0]
    pred = int(model.predict(x)[0])
    return pred, float(proba)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='Path to JSON file with feature values')
    parser.add_argument('--sample', action='store_true', help='Use a synthetic sample')
    args = parser.parse_args()

    model = joblib.load(MODELS / "best_model.joblib")
    columns = load_feature_names()

    if args.sample:
        # Create a simple sample using column means from the dataset
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        feat_means = data.data.mean().to_dict()
        pred, proba = predict_from_dict(model, feat_means)
        print("Prediction (0=malignant, 1=benign):", pred, "Probability of benign:", round(proba, 4))
        return

    if args.json:
        d = json.loads(Path(args.json).read_text(encoding='utf-8'))
        pred, proba = predict_from_dict(model, d)
        print("Prediction (0=malignant, 1=benign):", pred, "Probability of benign:", round(proba, 4))
        return

    print("Provide --sample or --json <path>")
    print("Required feature keys:", columns)

if __name__ == "__main__":
    main()
