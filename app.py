# import streamlit as st
# import pandas as pd
# from predict import predict_crop

# # Load dataset (for dropdown)
# df = pd.read_csv("./data/west_bengal_cleaned_crop.csv")

# # Clean for UI
# df["district"] = df["district"].str.lower().str.strip()
# df["season"] = df["season"].str.lower().str.strip()

# # Unique values
# districts = sorted(df["district"].unique())
# seasons = sorted(df["season"].unique())

# # ----------------------------
# # UI
# # ----------------------------
# st.set_page_config(page_title="Crop Recommendation", layout="centered")

# st.title("🌾 Crop Recommendation System (Baseline)")

# st.write("📌 Enter district and season to get crop recommendation")

# # Inputs
# district = st.selectbox("Select District", districts)
# season = st.selectbox("Select Season", seasons)

# # Button
# if st.button("🔍 Recommend Crop"):

#     result = predict_crop(district, season)

#     if isinstance(result, str):
#         st.error(result)
#     else:
#         st.success("✅ Prediction Generated")

#         st.subheader("🌳 Random Forest Result")
#         st.write(f"Crop: **{result['rf_crop']}**")
#         # st.write(f"Confidence: {result['rf_confidence']:.2f}")

#         st.subheader("🤖 Neural Network Result")
#         st.write(f"Crop: **{result['nn_crop']}**")
#         # st.write(f"Confidence: {result['nn_confidence']:.2f}")




############
# proposed #
###########



import streamlit as st
import pandas as pd
from predict import predict_improved

# Load dataset
df = pd.read_csv("./data/west_bengal_cleaned_crop.csv")

# Clean
df["district"] = df["district"].str.lower().str.strip()
df["season"] = df["season"].str.lower().str.strip()

# Apply same filtering
top_crops = df["crop"].value_counts().head(6).index
df = df[df["crop"].isin(top_crops)]

districts = sorted(df["district"].unique())
seasons = sorted(df["season"].unique())

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation System")

# st.write("Uses feature engineering + balanced dataset")

district = st.selectbox("Select District", districts)
season = st.selectbox("Select Season", seasons)

if st.button("🔍 Recommend Crop"):

    result = predict_improved(district, season)

    if isinstance(result, str):
        st.error(result)
    else:
        st.success("✅ Prediction Generated")

        st.subheader("🌳 Random Forest ")
        st.write(f"Crop: **{result['rf_crop']}**")
        # st.write(f"Confidence: {result['rf_confidence']:.2f}")

        st.subheader("🤖 Neural Network ")
        st.write(f"Crop: **{result['nn_crop']}**")
        # st.write(f"Confidence: {result['nn_confidence']:.2f}")