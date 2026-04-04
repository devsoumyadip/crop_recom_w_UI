import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# ----------------------------
# FILTER TOP CROPS
# ----------------------------
top_crops = real_df["crop"].value_counts().head(6).index
real_df = real_df[real_df["crop"].isin(top_crops)]
synthetic_df = synthetic_df[synthetic_df["crop"].isin(top_crops)]

# ----------------------------
# ENCODING (fit on real only)
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
# FEATURES
# ----------------------------
features = [
    "district", "season", "year",
    "area",
    "temperature", "rainfall", "humidity",
    "nitrogen", "phosphorus", "potassium",
    "organic_carbon", "ph", "micronutrient_score",
    "rainfall_temp_ratio", "npk_sum", "temp_humidity"
]

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------
def evaluate_model(y_test, pred, prob, name):

    acc = accuracy_score(y_test, pred)
    loss = log_loss(y_test, prob)
    mse = mean_squared_error(y_test, pred)

    try:
        r2 = r2_score(y_test, pred)
    except:
        r2 = 0

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Loss:", loss)
    print("MSE:", mse)
    print("R2 Score:", r2)


# ----------------------------
# TRAIN ALL MODELS
# ----------------------------
def train_all_models(df, label):

    X = df[features]
    y = df["crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n==============================")
    print(f"📊 {label}")
    print(f"==============================")

    # ----------------------------
    # RANDOM FOREST
    # ----------------------------
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    evaluate_model(
        y_test,
        rf.predict(X_test),
        rf.predict_proba(X_test),
        "RANDOM FOREST"
    )

    # ----------------------------
    # LIGHTGBM
    # ----------------------------
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    lgb.fit(X_train, y_train)

    evaluate_model(
        y_test,
        lgb.predict(X_test),
        lgb.predict_proba(X_test),
        "LIGHTGBM"
    )

    # ----------------------------
    # XGBOOST
    # ----------------------------
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    xgb.fit(X_train, y_train)

    evaluate_model(
        y_test,
        xgb.predict(X_test),
        xgb.predict_proba(X_test),
        "XGBOOST"
    )

    # ----------------------------
    # CATBOOST
    # ----------------------------
    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        verbose=0
    )

    cat.fit(X_train, y_train)

    evaluate_model(
        y_test,
        cat.predict(X_test),
        cat.predict_proba(X_test),
        "CATBOOST"
    )

    # ----------------------------
    # NEURAL NETWORK (MLP)
    # ----------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)

    nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train_nn.shape[1], activation='softmax')
    ])

    nn.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

    nn.fit(X_train_s, y_train_nn, epochs=50, batch_size=32, verbose=0)

    nn_prob = nn.predict(X_test_s, verbose=0)
    nn_pred = np.argmax(nn_prob, axis=1)

    evaluate_model(y_test, nn_pred, nn_prob, "MLP (NN)")

    return lgb


# =====================================================
# 🧪 EXPERIMENTS
# =====================================================

# 1. REAL ONLY
model_real = train_all_models(real_df, "REAL DATA ONLY")

# 2. REAL + 50% SYNTHETIC
synthetic_50 = synthetic_df.sample(frac=0.5, random_state=42)
df_50 = pd.concat([real_df, synthetic_50], ignore_index=True)

model_50 = train_all_models(df_50, "REAL + 50% SYNTHETIC")

# 3. REAL + 100% SYNTHETIC
df_full = pd.concat([real_df, synthetic_df], ignore_index=True)

model_full = train_all_models(df_full, "REAL + 100% SYNTHETIC")

# ----------------------------
# SAVE BEST MODEL
# ----------------------------
joblib.dump(model_full, "models/lgb_final.pkl")

joblib.dump({
    "district": le_district,
    "season": le_season,
    "crop": le_crop
}, "models/encoders.pkl")

print("\n✅ FULL TRAINING COMPLETE")