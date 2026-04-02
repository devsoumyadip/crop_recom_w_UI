import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap

# ----------------------------
# LOAD MODELS
# ----------------------------
rf_model = joblib.load("models/rf_improved.pkl")
nn_model = tf.keras.models.load_model("models/nn_improved.h5")
lgb_model = joblib.load("models/lgb_model.pkl")

scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

le_district = encoders["district"]
le_season = encoders["season"]
le_crop = encoders["crop"]

# SHAP (use best model)
explainer = shap.TreeExplainer(lgb_model)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season", "crop"]:
    df[col] = df[col].str.lower().str.strip()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def prepare_input(district, season):

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

    d = le_district.transform([district])[0]
    s = le_season.transform([season])[0]
    year = df["year"].max()

    input_df = pd.DataFrame([{
        "district": d,
        "season": s,
        "year": year,
        **avg_vals.to_dict()
    }])

    return input_df


# ----------------------------
# SHAP
# ----------------------------
def get_shap_explanation(input_df):

    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        pred_class = np.argmax(lgb_model.predict_proba(input_df))
        shap_vals = shap_values[pred_class]
    else:
        shap_vals = shap_values

    shap_vals = np.array(shap_vals).reshape(-1)

    explanation = sorted(
        zip(input_df.columns, shap_vals),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )

    return explanation[:5]


# ----------------------------
# 🚀 FULL PREDICT
# ----------------------------
def predict_full(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    input_df = prepare_input(district, season)

    # ----------------------------
    # MODEL PREDICTIONS
    # ----------------------------
    rf_probs = rf_model.predict_proba(input_df)[0]

    scaled = scaler.transform(input_df)
    nn_probs = nn_model.predict(scaled, verbose=0)[0]

    lgb_probs = lgb_model.predict_proba(input_df)[0]

    # ----------------------------
    # 🔥 ENSEMBLE
    # ----------------------------
    ensemble_probs = (0.2 * rf_probs + 0.2 * nn_probs + 0.6 * lgb_probs)

    # ----------------------------
    # 🎯 TOP 1 (LGBM)
    # ----------------------------
    lgb_idx = np.argmax(lgb_probs)
    lgb_crop = le_crop.inverse_transform([lgb_idx])[0]
    lgb_conf = lgb_probs[lgb_idx]

    # ----------------------------
    # 🎯 TOP 1 (ENSEMBLE)
    # ----------------------------
    ens_idx = np.argmax(ensemble_probs)
    ens_crop = le_crop.inverse_transform([ens_idx])[0]
    ens_conf = ensemble_probs[ens_idx]

    # ----------------------------
    # 🧠 SHAP
    # ----------------------------
    shap_values = get_shap_explanation(input_df)

    return {
        "lgbm": (lgb_crop, float(lgb_conf)),
        "ensemble": (ens_crop, float(ens_conf)),
        "shap": shap_values
    }