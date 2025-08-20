import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🏥 Breast Cancer Malignancy Prediction")
st.write("Input diagnostic features to predict whether a tumor is **malignant (0)** or **benign (1)**.")

@st.cache_data
def get_feature_schema():
    data = load_breast_cancer(as_frame=True)
    return list(data.feature_names), data.data.describe()

def build_inputs(feature_names, desc: pd.DataFrame):
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(feature_names):
        col = cols[i % 3]
        mean = float(desc.loc['mean', feat])
        std = float(desc.loc['std', feat])
        minv = float(desc.loc['min', feat])
        maxv = float(desc.loc['max', feat])
        values[feat] = col.number_input(
            feat, value=mean, min_value=minv, max_value=maxv, step=std/10 if std>0 else 0.01, format="%.4f"
        )
    return values

feature_names, stats = get_feature_schema()
inputs = build_inputs(feature_names, stats)

if not MODEL_PATH.exists():
    st.warning("Best model not found. Please run `python src/train.py` first.")
else:
    model = joblib.load(MODEL_PATH)
    x = np.array([[inputs[f] for f in feature_names]], dtype=float)
    proba = model.predict_proba(x)[:, 1][0]
    pred = int(model.predict(x)[0])

    st.markdown("---")
    st.subheader("Prediction")
    st.metric("Class (0=malignant, 1=benign)", pred)
    st.metric("Probability of benign", f"{proba:.4f}")
