import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap

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
# SHAP EXPLAINER (for RF Improved)
# ----------------------------
explainer = shap.TreeExplainer(rf_improved)

# ----------------------------
# LOAD DATA (for avg values)
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
        "area",
        "temperature", "rainfall", "humidity",
        "nitrogen", "phosphorus", "potassium",
        "organic_carbon", "ph", "micronutrient_score"
    ]

    avg_vals = subset[cols].mean()

    if avg_vals.isnull().any():
        avg_vals = df[cols].mean()

    return avg_vals


# ----------------------------
# 🧠 SHAP FUNCTION
# ----------------------------
def get_shap_explanation(input_df):

    shap_values = explainer.shap_values(input_df)

    # ----------------------------
    # HANDLE DIFFERENT SHAP FORMATS
    # ----------------------------
    if isinstance(shap_values, list):
        # multiclass → list of arrays
        pred_class = np.argmax(rf_improved.predict_proba(input_df))
        shap_vals = shap_values[pred_class]

        # ensure 1D
        shap_vals = np.array(shap_vals).reshape(-1)

    else:
        # single output
        shap_vals = np.array(shap_values)

        # flatten safely
        shap_vals = shap_vals.reshape(-1)

    feature_names = input_df.columns

    # ----------------------------
    # SAFE SORT
    # ----------------------------
    explanation = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(float(x[1])),  # 🔥 force scalar
        reverse=True
    )

    return explanation[:5]
# ----------------------------
# 🧪 BASELINE PREDICTION
# ----------------------------
def predict_baseline(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

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

    # RF
    rf_probs = rf_baseline.predict_proba(input_df)[0]
    rf_idx = np.argmax(rf_probs)
    rf_crop = le_crop.inverse_transform([rf_idx])[0]
    rf_conf = rf_probs[rf_idx]

    # NN
    nn_probs = nn_baseline.predict(input_df, verbose=0)[0]
    nn_idx = np.argmax(nn_probs)
    nn_crop = le_crop.inverse_transform([nn_idx])[0]
    nn_conf = nn_probs[nn_idx]

    return {
        "rf": (rf_crop, float(rf_conf)),
        "nn": (nn_crop, float(nn_conf))
    }


# ----------------------------
# 🚀 IMPROVED PREDICTION
# ----------------------------
def predict_improved(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    avg_vals = get_avg_features(district)

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

    # RF
    rf_probs = rf_improved.predict_proba(input_df)[0]
    rf_idx = np.argmax(rf_probs)
    rf_crop = le_crop.inverse_transform([rf_idx])[0]
    rf_conf = rf_probs[rf_idx]

    # NN
    scaled = scaler.transform(input_df)
    nn_probs = nn_improved.predict(scaled, verbose=0)[0]
    nn_idx = np.argmax(nn_probs)
    nn_crop = le_crop.inverse_transform([nn_idx])[0]
    nn_conf = nn_probs[nn_idx]

    # SHAP
    shap_values = get_shap_explanation(input_df)

    return {
        "rf": (rf_crop, float(rf_conf)),
        "nn": (nn_crop, float(nn_conf)),
        "shap": shap_values
    }