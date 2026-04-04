import pandas as pd
import numpy as np
import joblib
import shap

# ----------------------------
# LOAD MODELS
# ----------------------------
xgb_model = joblib.load("models/xgb_final.pkl")

encoders = joblib.load("models/encoders.pkl")

le_district = encoders["district"]
le_season = encoders["season"]
le_crop = encoders["crop"]

# SHAP (use XGBoost)
explainer = shap.TreeExplainer(xgb_model)

# ----------------------------
# LOAD DATA (for feature engineering)
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season", "crop"]:
    df[col] = df[col].astype(str).str.lower().str.strip()


# ----------------------------
# FEATURE ENGINEERING (SAME AS TRAIN)
# ----------------------------
def prepare_input(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    subset = df[df["district"] == district]

    cols = [
        "area",
        "temperature", "rainfall", "humidity",
        "nitrogen", "phosphorus", "potassium",
        "organic_carbon", "ph", "micronutrient_score"
    ]

    avg_vals = subset[cols].mean()

    # fallback
    if avg_vals.isnull().any():
        avg_vals = df[cols].mean()

    # encode
    d = le_district.transform([district])[0]
    s = le_season.transform([season])[0]
    year = df["year"].max()

    input_df = pd.DataFrame([{
        "district": d,
        "season": s,
        "year": year,
        **avg_vals.to_dict()
    }])

    # ----------------------------
    # ADD ENGINEERED FEATURES
    # ----------------------------
    input_df["rainfall_temp_ratio"] = input_df["rainfall"] / (input_df["temperature"] + 1)
    input_df["npk_sum"] = input_df["nitrogen"] + input_df["phosphorus"] + input_df["potassium"]
    input_df["temp_humidity"] = input_df["temperature"] * input_df["humidity"]
    input_df["rainfall_per_area"] = input_df["rainfall"] / (input_df["area"] + 1)

    return input_df


# ----------------------------
# SHAP EXPLANATION
# ----------------------------
def get_shap_explanation(input_df):

    shap_values = explainer.shap_values(input_df)

    # handle multi-class
    if isinstance(shap_values, list):
        pred_class = np.argmax(xgb_model.predict_proba(input_df))
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
# FINAL PREDICTION FUNCTION
# ----------------------------
def predict_crop(district, season):

    district = district.lower().strip()
    season = season.lower().strip()

    if district not in le_district.classes_:
        return f"❌ District '{district}' not found"

    if season not in le_season.classes_:
        return f"❌ Season '{season}' not found"

    input_df = prepare_input(district, season)

    # ----------------------------
    # XGBOOST PREDICTION
    # ----------------------------
    probs = xgb_model.predict_proba(input_df)[0]

    pred_idx = np.argmax(probs)

    crop = le_crop.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # ----------------------------
    # SHAP
    # ----------------------------
    shap_values = get_shap_explanation(input_df)

    return {
        "crop": crop,
        "confidence": confidence,
        "shap": shap_values
    }