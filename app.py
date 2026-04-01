import streamlit as st
import pandas as pd
from predict import predict_baseline, predict_improved

# ----------------------------
# LOAD DATA (for dropdowns)
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season"]:
    df[col] = df[col].str.lower().str.strip()

districts = sorted(df["district"].unique())
seasons = sorted(df["season"].unique())

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.markdown("### Compare Baseline vs Improved Models")

# ----------------------------
# INPUTS
# ----------------------------
district = st.selectbox("📍 Select District", districts)
season = st.selectbox("🌦️ Select Season", seasons)

# ----------------------------
# BUTTON
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
        # 🧪 BASELINE MODELS
        # ============================
        st.markdown("## 🧪 Baseline Models")

        # RF Baseline (Top-1)
        rf_index_base = baseline["rf_probs"].argmax()
        rf_crop_base = baseline["top3"][0][0]
        rf_conf_base = baseline["rf_probs"][rf_index_base]

        st.subheader("🌳 Random Forest (Baseline)")
        st.write(f"**Prediction:** {rf_crop_base}")
        # st.write(f"Confidence: {rf_conf_base:.3f}")

        # NN Baseline (Top-1)
        nn_index_base = baseline["nn_probs"].argmax()
        nn_crop_base = baseline["top3"][0][0]
        nn_conf_base = baseline["nn_probs"][nn_index_base]

        st.subheader("🤖 Neural Network (Baseline)")
        st.write(f"**Prediction:** {nn_crop_base}")
        # st.write(f"Confidence: {nn_conf_base:.3f}")

        # ============================
        # 🚀 IMPROVED MODELS
        # ============================
        st.markdown("## 🚀 Improved Models")

        # RF Improved (Top-1)
        rf_index_imp = improved["rf_probs"].argmax()
        rf_crop_imp = improved["top3"][0][0]
        rf_conf_imp = improved["rf_probs"][rf_index_imp]

        st.subheader("🌳 Random Forest (Improved)")
        st.write(f"**Prediction:** {rf_crop_imp}")
        # st.write(f"Confidence: {rf_conf_imp:.3f}")

        # NN Improved (Top-1)
        nn_index_imp = improved["nn_probs"].argmax()
        nn_crop_imp = improved["top3"][0][0]
        nn_conf_imp = improved["nn_probs"][nn_index_imp]

        st.subheader("🤖 Neural Network (Improved)")
        st.write(f"**Prediction:** {nn_crop_imp}")
        # st.write(f"Confidence: {nn_conf_imp:.3f}")