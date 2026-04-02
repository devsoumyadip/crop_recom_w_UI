import streamlit as st
import pandas as pd
from predict import predict_full

# Load data
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season"]:
    df[col] = df[col].str.lower().str.strip()

districts = sorted(df["district"].unique())
seasons = sorted(df["season"].unique())

# UI
st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.title("🌾 Smart Crop Recommendation (AI Pipeline)")
st.markdown("### LightGBM + RF + NN + Explainable AI")

district = st.selectbox("📍 District", districts)
season = st.selectbox("🌦️ Season", seasons)

if st.button("🚀 Predict"):

    result = predict_full(district, season)

    if isinstance(result, str):
        st.error(result)
    else:
        st.success("✅ Prediction Generated")

        # ----------------------------
        # TOP 3 CROPS
        # ----------------------------
        st.markdown("## 🌱 Recommended Crops")

        for i, (crop, prob) in enumerate(result["top3"], 1):
            st.write(f"{i}. **{crop}** → {prob:.3f}")

        # ----------------------------
        # SHAP
        # ----------------------------
        st.markdown("## 🧠 Why this prediction?")

        for feature, value in result["shap"]:
            impact = "⬆️ increases" if value > 0 else "⬇️ decreases"
            st.write(f"{feature} → {impact} prediction ({value:.3f})")