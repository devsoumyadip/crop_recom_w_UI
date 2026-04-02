import streamlit as st
import pandas as pd
from predict import predict_baseline, predict_improved

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
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("### Baseline vs Improved Models")

district = st.selectbox("📍 Select District", districts)
season = st.selectbox("🌦️ Select Season", seasons)

# ----------------------------
# PREDICT
# ----------------------------
if st.button("🚀 Predict Crop"):

    baseline = predict_baseline(district, season)
    improved = predict_improved(district, season)

    if isinstance(baseline, str):
        st.error(baseline)

    elif isinstance(improved, str):
        st.error(improved)

    else:
        st.success("✅ Prediction Generated")

        # ============================
        # BASELINE
        # ============================
        st.markdown("## 🧪 Baseline Models")

        st.subheader("🌳 Random Forest (Baseline)")
        st.write(f"**Prediction:** {baseline['rf'][0]}")
        # st.write(f"Confidence: {baseline['rf'][1]:.3f}")

        st.subheader("🤖 Neural Network (Baseline)")
        st.write(f"**Prediction:** {baseline['nn'][0]}")
        # st.write(f"Confidence: {baseline['nn'][1]:.3f}")

        # ============================
        # IMPROVED
        # ============================
        st.markdown("## 🚀 Improved Models")

        st.subheader("🌳 Random Forest (Improved)")
        st.write(f"**Prediction:** {improved['rf'][0]}")
        # st.write(f"Confidence: {improved['rf'][1]:.3f}")

        st.subheader("🤖 Neural Network (Improved)")
        st.write(f"**Prediction:** {improved['nn'][0]}")
        # st.write(f"Confidence: {improved['nn'][1]:.3f}")

        # ============================
        # SHAP
        # ============================
        st.markdown("## 🧠 Why this prediction? (Explainable AI)")

        for feature, value in improved["shap"]:
            impact = "⬆️ increases" if value > 0 else "⬇️ decreases"
            st.write(f"{feature} → {impact} prediction ({value:.3f})")