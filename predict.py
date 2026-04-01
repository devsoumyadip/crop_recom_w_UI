# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf

# # ----------------------------
# # LOAD MODELS
# # ----------------------------
# rf_model = joblib.load("models/rf_model.pkl")
# nn_model = tf.keras.models.load_model("models/nn_model.h5")
# encoders = joblib.load("models/encoders.pkl")

# le_district = encoders["district"]
# le_season = encoders["season"]
# le_crop = encoders["crop"]

# # ----------------------------
# # PREDICTION FUNCTION
# # ----------------------------
# def predict_crop(district, season):

#     # Clean input
#     district = district.lower().strip()
#     season = season.lower().strip()

#     # Validate input
#     if district not in le_district.classes_:
#         return f"❌ District '{district}' not found in dataset"

#     if season not in le_season.classes_:
#         return f"❌ Season '{season}' not found in dataset"

#     # Encode
#     d = le_district.transform([district])[0]
#     s = le_season.transform([season])[0]

#     # Create input (ONLY 2 features as per paper)
#     input_df = pd.DataFrame([{
#         "district": d,
#         "season": s
#     }])

#     # ----------------------------
#     # 🌳 RANDOM FOREST
#     # ----------------------------
#     rf_pred = rf_model.predict(input_df)
#     rf_crop = le_crop.inverse_transform(rf_pred)[0]

#     rf_probs = rf_model.predict_proba(input_df)[0]
#     rf_conf = np.max(rf_probs)

#     # ----------------------------
#     # 🤖 NEURAL NETWORK
#     # ----------------------------
#     nn_pred = nn_model.predict(input_df, verbose=0)
#     nn_class = np.argmax(nn_pred)
#     nn_crop = le_crop.inverse_transform([nn_class])[0]
#     nn_conf = np.max(nn_pred)

#     # ----------------------------
#     # OUTPUT
#     # ----------------------------
#     return {
#         "rf_crop": rf_crop,
#         "rf_confidence": float(rf_conf),
#         "nn_crop": nn_crop,
#         "nn_confidence": float(nn_conf)
#     }



############
# proposed #
###########


import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ----------------------------
# LOAD MODELS
# ----------------------------
rf_model = joblib.load("models/rf_model_improved.pkl")
scaler = joblib.load("models/scaler_improved.pkl")
encoders = joblib.load("models/encoders_improved.pkl")
nn_model = tf.keras.models.load_model("models/nn_model_improved.h5")

le_district = encoders["district"]
le_season = encoders["season"]
le_crop = encoders["crop"]

# Load dataset (for avg values)
df = pd.read_csv("./data/west_bengal_cleaned_crop.csv")


# Clean
df.columns = df.columns.str.lower().str.strip()
for col in ["district", "season", "crop"]:
    df[col] = df[col].str.lower().str.strip()

# Apply same filtering as training
top_crops = df["crop"].value_counts().head(6).index
df = df[df["crop"].isin(top_crops)]

# Feature engineering (same as training)
df["productivity"] = df["production"] / (df["area"] + 1)


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_improved(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    # Validate
    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    subset = df[df["district"] == district]

    # Use average values
    avg_vals = subset[["area", "production", "productivity"]].mean()

    # Fallback if missing
    if avg_vals.isnull().any():
        avg_vals = df[["area", "production", "productivity"]].mean()

    # Encode
    d = le_district.transform([district])[0]
    s = le_season.transform([season])[0]

    year = df["year"].max()

    input_df = pd.DataFrame([{
        "district": d,
        "season": s,
        "year": year,
        "area": avg_vals["area"],
        "production": avg_vals["production"],
        "productivity": avg_vals["productivity"]
    }])

    # ----------------------------
    # 🌳 RANDOM FOREST
    # ----------------------------
    rf_pred = rf_model.predict(input_df)
    rf_crop = le_crop.inverse_transform(rf_pred)[0]

    rf_probs = rf_model.predict_proba(input_df)[0]
    rf_conf = np.max(rf_probs)

    # ----------------------------
    # 🤖 NEURAL NETWORK
    # ----------------------------
    scaled = scaler.transform(input_df)

    nn_pred = nn_model.predict(scaled, verbose=0)
    nn_class = np.argmax(nn_pred)
    nn_crop = le_crop.inverse_transform([nn_class])[0]
    nn_conf = np.max(nn_pred)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    return {
        "rf_crop": rf_crop,
        "rf_confidence": float(rf_conf),
        "nn_crop": nn_crop,
        "nn_confidence": float(nn_conf)
    }