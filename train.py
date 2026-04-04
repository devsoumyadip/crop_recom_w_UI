import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ----------------------------
# LOAD DATA
# ----------------------------
real_df = pd.read_csv("./data/finalData.csv")
synthetic_df = pd.read_csv("./data/synthetic_data.csv")

# CLEAN
for df in [real_df, synthetic_df]:
    df.columns = df.columns.str.lower().str.strip()
    df.drop(columns=["unnamed: 0"], inplace=True, errors="ignore")

    for col in ["district", "season", "crop"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
for df in [real_df, synthetic_df]:
    df["rainfall_temp_ratio"] = df["rainfall"] / (df["temperature"] + 1)
    df["npk_sum"] = df["nitrogen"] + df["phosphorus"] + df["potassium"]
    df["temp_humidity"] = df["temperature"] * df["humidity"]
    df["rainfall_per_area"] = df["rainfall"] / (df["area"] + 1)

# ----------------------------
# FILTER TOP CROPS
# ----------------------------
top_crops = real_df["crop"].value_counts().head(6).index
real_df = real_df[real_df["crop"].isin(top_crops)]
synthetic_df = synthetic_df[synthetic_df["crop"].isin(top_crops)]

# ----------------------------
# ENCODING
# ----------------------------
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

real_df["district"] = le_district.fit_transform(real_df["district"])
real_df["season"] = le_season.fit_transform(real_df["season"])
real_df["crop"] = le_crop.fit_transform(real_df["crop"])

synthetic_df["district"] = le_district.transform(synthetic_df["district"])
synthetic_df["season"] = le_season.transform(synthetic_df["season"])
synthetic_df["crop"] = le_crop.transform(synthetic_df["crop"])

# ----------------------------
# COMBINE DATA (100% synthetic)
# ----------------------------
df = pd.concat([real_df, synthetic_df], ignore_index=True)

print("Final dataset size:", df.shape)

# ----------------------------
# FEATURES
# ----------------------------
features = [
    "district", "season", "year",
    "area",
    "temperature", "rainfall", "humidity",
    "nitrogen", "phosphorus", "potassium",
    "organic_carbon", "ph", "micronutrient_score",
    "rainfall_temp_ratio", "npk_sum", "temp_humidity", "rainfall_per_area"
]

X = df[features]
y = df["crop"]

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 🌳 LIGHTGBM
# =====================================================
lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)
lgb_prob = lgb_model.predict_proba(X_test)

print("\n===== LIGHTGBM =====")
print("Accuracy:", accuracy_score(y_test, lgb_pred))
print("Loss:", log_loss(y_test, lgb_prob))

# =====================================================
# ⚡ XGBOOST
# =====================================================
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)

print("\n===== XGBOOST =====")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Loss:", log_loss(y_test, xgb_prob))

# =====================================================
# 🔥 ENSEMBLE (FINAL MODEL)
# =====================================================
ensemble_prob = 0.55 * lgb_prob + 0.45 * xgb_prob
ensemble_pred = np.argmax(ensemble_prob, axis=1)

print("\n===== LGBM + XGB ENSEMBLE =====")
print("Accuracy:", accuracy_score(y_test, ensemble_pred))
print("Loss:", log_loss(y_test, ensemble_prob))

# ----------------------------
# SAVE MODELS
# ----------------------------
joblib.dump(lgb_model, "models/lgb_final.pkl")
joblib.dump(xgb_model, "models/xgb_final.pkl")

joblib.dump({
    "district": le_district,
    "season": le_season,
    "crop": le_crop
}, "models/encoders.pkl")

print("\n✅ FINAL ENSEMBLE TRAINING COMPLETE")