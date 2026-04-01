import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ----------------------------
# LOAD MODELS
# ----------------------------
rf_baseline = joblib.load("models/rf_baseline.pkl")
nn_baseline = tf.keras.models.load_model("models/nn_baseline.h5")

rf_improved = joblib.load("models/rf_improved.pkl")
nn_improved = tf.keras.models.load_model("models/nn_improved.h5")

scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

le_district = encoders["district"]
le_season = encoders["season"]
le_crop = encoders["crop"]

# ----------------------------
# LOAD DATA (for averages)
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season", "crop"]:
    df[col] = df[col].str.lower().str.strip()

# ----------------------------
# HELPER: AVG FEATURES
# ----------------------------
def get_avg_features(district):

    subset = df[df["district"] == district]

    cols = [
        "area", "production", "yield",
        "temperature", "rainfall", "humidity",
        "nitrogen", "phosphorus", "potassium",
        "organic_carbon", "ph", "micronutrient_score"
    ]

    avg_vals = subset[cols].mean()

    if avg_vals.isnull().any():
        avg_vals = df[cols].mean()

    return avg_vals


# ----------------------------
# 🧪 BASELINE PREDICTION
# ----------------------------
def predict_baseline(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    # Validation
    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    d = le_district.transform([district])[0]
    s = le_season.transform([season])[0]

    input_df = pd.DataFrame([{
        "district": d,
        "season": s
    }])

    # 🌳 RF
    rf_probs = rf_baseline.predict_proba(input_df)[0]

    # 🤖 NN (no scaling)
    nn_probs = nn_baseline.predict(input_df, verbose=0)[0]

    # 🔥 Ensemble (optional but good)
    final_probs = (rf_probs + nn_probs) / 2

    # 🔝 TOP 3
    top3_idx = np.argsort(final_probs)[::-1][:3]

    crops = le_crop.inverse_transform(top3_idx)
    probs = final_probs[top3_idx]

    return {
        "top3": list(zip(crops, probs)),
        "rf_probs": rf_probs,
        "nn_probs": nn_probs
    }


# ----------------------------
# 🚀 IMPROVED PREDICTION
# ----------------------------
def predict_improved(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    # Validation
    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    # Get average environmental values
    avg_vals = get_avg_features(district)

    # Encode
    d = le_district.transform([district])[0]
    s = le_season.transform([season])[0]

    year = df["year"].max()

    input_df = pd.DataFrame([{
        "district": d,
        "season": s,
        "year": year,
        "area": avg_vals["area"],
        "temperature": avg_vals["temperature"],
        "rainfall": avg_vals["rainfall"],
        "humidity": avg_vals["humidity"],
        "nitrogen": avg_vals["nitrogen"],
        "phosphorus": avg_vals["phosphorus"],
        "potassium": avg_vals["potassium"],
        "organic_carbon": avg_vals["organic_carbon"],
        "ph": avg_vals["ph"],
        "micronutrient_score": avg_vals["micronutrient_score"]
    }])

    # 🌳 RF
    rf_probs = rf_improved.predict_proba(input_df)[0]

    # 🤖 NN (scaled)
    scaled = scaler.transform(input_df)
    nn_probs = nn_improved.predict(scaled, verbose=0)[0]

    # 🔥 Ensemble
    final_probs = (rf_probs + nn_probs) / 2

    # 🔝 TOP 3
    top3_idx = np.argsort(final_probs)[::-1][:3]

    crops = le_crop.inverse_transform(top3_idx)
    probs = final_probs[top3_idx]

    return {
        "top3": list(zip(crops, probs)),
        "rf_probs": rf_probs,
        "nn_probs": nn_probs
    }


# ----------------------------
# 🎯 COMBINED FUNCTION (FOR UI)
# ----------------------------
def predict_all(district, season):

    base = predict_baseline(district, season)
    imp = predict_improved(district, season)

    if isinstance(base, str):
        return base

    if isinstance(imp, str):
        return imp

    return {
        "baseline": base,
        "improved": imp
    }