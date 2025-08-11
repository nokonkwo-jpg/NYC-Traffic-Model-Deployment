from pathlib import Path
import os, io
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Traffic Volume Predictor", page_icon="🚗", layout="centered")

# ========= Model files & storage =========
def _pick_models_dir() -> Path:
    # prefer ./models; if not writable (e.g., some container hosts), fallback to /tmp/models
    d = Path("models")
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".writetest").write_text("ok", encoding="utf-8")
        (d / ".writetest").unlink(missing_ok=True)
        return d
    except Exception:
        d = Path("/tmp/models")
        d.mkdir(parents=True, exist_ok=True)
        return d

MODELS_DIR = _pick_models_dir()

MODEL_FILES = {
    "hgb": MODELS_DIR / "hgb_model.joblib",
    "rf":  MODELS_DIR / "rf_model.joblib",
    "seg": MODELS_DIR / "segmented_model.joblib",
}

# Hard-coded Google Drive links (add more when you have them)
GDRIVE_URLS = {
    "hgb": "https://drive.google.com/file/d/17DP_Cd4v4MYqkU5sQPOTNqNSAo2riPdL/view?usp=sharing",
    "rf":  "https://drive.google.com/file/d/1ICsgr0GNTDtLQ8JVAxjljsalAmI7qU4g/view?usp=sharing", 
    "seg": "https://drive.google.com/file/d/1AqzUP7ND3cfLddi13QD7l3REgruKYo_u/view?usp=sharing", 
}

FEATURES = [
    "hour_sin","hour_cos","wd_sin","wd_cos",
    "month_sin","month_cos","vol_lag_1","vol_roll_3h","vol_roll_24h",
]

def _download_if_missing(url: str, dest: Path) -> bool:
    """Return True if file exists after this call (downloaded or already present)."""
    if dest.exists():
        return True
    if not url:
        return False
    import gdown
    try:
        with st.spinner(f"Downloading {dest.name}…"):
            # fuzzy=True lets us pass a 'view' URL or file id
            gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        return dest.exists()
    except Exception as e:
        st.error(f"Download failed for {dest.name}: {e}")
        return False

@st.cache_resource(show_spinner=True)
def prepare_and_load_models():
    # Always try to fetch first (so “not available locally” gets fixed)
    for key, dest in MODEL_FILES.items():
        _download_if_missing(GDRIVE_URLS.get(key), dest)

    loaded = {}
    for key, dest in MODEL_FILES.items():
        if dest.exists():
            try:
                loaded[key] = joblib.load(dest)
            except Exception as e:
                st.error(f"Failed to load {dest.name}: {e}")

    if not loaded:
        st.stop()  # show errors above and stop the app
    return loaded

MODELS = prepare_and_load_models()
pretty = {"hgb": "HistGradientBoosting", "rf": "RandomForest", "seg": "Segmented"}

st.title("🚗 Traffic Volume Predictor")
st.caption("Downloads models first, then predicts. Add RF/SEG Drive links to enable those.")

# Optional: manual re-download button (e.g., if you updated the files in Drive)
if st.button("🔁 Re-download models"):
    for k, p in MODEL_FILES.items():
        if GDRIVE_URLS.get(k):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    # Clear the cache and re-run
    prepare_and_load_models.clear()
    st.experimental_rerun()

# Model picker (only those we have files for)
model_key = st.selectbox("Model", options=list(MODELS.keys()), format_func=lambda k: pretty.get(k, k))
model = MODELS[model_key]

mode = st.radio("Input mode", ["Sliders (single row)", "CSV upload"], horizontal=True)

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in FEATURES if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    y = model.predict(df[FEATURES])
    # If you trained on log1p, uncomment the next line:
    # y = np.expm1(y)
    return pd.DataFrame({"prediction": y}, index=df.index)

if mode == "Sliders (single row)":
    st.subheader("Features")
    c1, c2, c3 = st.columns(3)

    # bounded sines/cosines
    hour_sin  = c1.slider("hour_sin",  -1.0, 1.0, 0.00, 0.01)
    hour_cos  = c2.slider("hour_cos",  -1.0, 1.0, 1.00, 0.01)
    wd_sin    = c3.slider("wd_sin",    -1.0, 1.0, 0.00, 0.01)
    wd_cos    = c1.slider("wd_cos",    -1.0, 1.0, 1.00, 0.01)
    month_sin = c2.slider("month_sin", -1.0, 1.0, 0.50, 0.01)
    month_cos = c3.slider("month_cos", -1.0, 1.0, 0.866, 0.001)

    # volumes — set wide but sane ranges; tweak as needed
    vol_lag_1    = c1.number_input("vol_lag_1",    min_value=0.0, max_value=1_000_000.0, value=100.0, step=1.0)
    vol_roll_3h  = c2.number_input("vol_roll_3h",  min_value=0.0, max_value=1_000_000.0, value=110.0, step=1.0)
    vol_roll_24h = c3.number_input("vol_roll_24h", min_value=0.0, max_value=1_000_000.0, value=115.0, step=1.0)

    if st.button("Predict"):
        row = {
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "wd_sin": wd_sin, "wd_cos": wd_cos,
            "month_sin": month_sin, "month_cos": month_cos,
            "vol_lag_1": vol_lag_1, "vol_roll_3h": vol_roll_3h, "vol_roll_24h": vol_roll_24h,
        }
        df = pd.DataFrame([row], columns=FEATURES)
        res = predict_df(df)
        st.success("Prediction")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download CSV", res.to_csv(index=False), "prediction.csv", "text/csv")

else:
    up = st.file_uploader("Upload CSV with columns:", type=["csv"])
    st.code(", ".join(FEATURES))
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("Preview:", df.head())
            if st.button("Predict"):
                res = predict_df(df)
                st.success("Predictions")
                st.dataframe(res, use_container_width=True)
                st.download_button("Download CSV", res.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(str(e))
