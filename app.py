import streamlit as st
import pandas as pd
from predict import predict_full

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season"]:
    df[col] = df[col].str.lower().str.strip()

districts = sorted(df["district"].unique())
seasons = sorted(df["season"].unique())

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("### LightGBM + Ensemble + Explainable AI")

district = st.selectbox("📍 Select District", districts)
season = st.selectbox("🌦️ Select Season", seasons)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("🚀 Predict Crop"):

    result = predict_full(district, season)

    if isinstance(result, str):
        st.error(result)

    else:
        st.success("✅ Prediction Generated")

        # ============================
        # LIGHTGBM
        # ============================
        st.markdown("## ⭐ LightGBM Prediction")

        st.write(f"**Crop:** {result['lgbm'][0]}")
        # st.write(f"Confidence: {result['lgbm'][1]:.3f}")

        # ============================
        # ENSEMBLE
        # ============================
        st.markdown("## 🔥 Ensemble Prediction")

        st.write(f"**Crop:** {result['ensemble'][0]}")
        # st.write(f"Confidence: {result['ensemble'][1]:.3f}")

        # ============================
        # SHAP
        # ============================
        st.markdown("## 🧠 Why this prediction? (Explainable AI)")

        for feature, value in result["shap"]:
            impact = "⬆️ increases" if value > 0 else "⬇️ decreases"
            st.write(f"{feature} → {impact} prediction ({value:.3f})")