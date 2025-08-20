from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_and_export_dataset(csv_path='data/raw/breast_cancer.csv'):
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Ensure target column is named 'target' and is binary (0=malignant, 1=benign) as per sklearn
    # sklearn's load_breast_cancer uses target: 0=malignant, 1=benign
    df.to_csv(csv_path, index=False)
    return df, data

if __name__ == "__main__":
    df, meta = load_and_export_dataset()
    print(f"Exported dataset with shape {df.shape}")
