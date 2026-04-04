import streamlit as st
import pandas as pd
from predict import predict_crop

# ----------------------------
# LOAD DATA (for dropdowns)
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

districts = sorted(df["district"].unique())
seasons = sorted(df["season"].unique())

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="centered"
)

# ----------------------------
# TITLE
# ----------------------------
st.title("🌾 Smart Crop Recommendation System")
st.markdown("### XGBoost + GAN + Explainable AI")

st.markdown("---")

# ----------------------------
# INPUT SECTION
# ----------------------------
st.subheader("📍 Select Inputs")

district = st.selectbox("Select District", districts)
season = st.selectbox("Select Season", seasons)

st.markdown("---")

# ----------------------------
# PREDICT BUTTON
# ----------------------------
if st.button("🚀 Predict Crop"):

    result = predict_crop(district, season)

    if isinstance(result, str):
        st.error(result)

    else:
        st.success("✅ Prediction Generated")

        # ----------------------------
        # RESULT
        # ----------------------------
        st.markdown("## 🌱 Recommended Crop")

        st.markdown(f"### **{result['crop'].upper()}**")
        st.write(f"**Confidence:** {result['confidence']:.3f}")

        st.markdown("---")

        # ----------------------------
        # SHAP EXPLANATION
        # ----------------------------
        st.markdown("## 🧠 Why this prediction?")

        for feature, value in result["shap"]:
            if value > 0:
                st.write(f"✔ **{feature}** → increases prediction ({value:.3f})")
            else:
                st.write(f"❌ **{feature}** → decreases prediction ({value:.3f})")

        st.markdown("---")

        # ----------------------------
        # FOOTER INSIGHT
        # ----------------------------
        st.info("This recommendation is based on environmental and soil conditions using AI models.")