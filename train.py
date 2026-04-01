import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

# Clean text
for col in ["district", "season", "crop"]:
    df[col] = df[col].str.lower().str.strip()

# ----------------------------
# FILTER TOP CROPS
# ----------------------------
top_crops = df["crop"].value_counts().head(6).index
df = df[df["crop"].isin(top_crops)]

# ----------------------------
# ENCODING
# ----------------------------
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df["district"] = le_district.fit_transform(df["district"])
df["season"] = le_season.fit_transform(df["season"])
df["crop"] = le_crop.fit_transform(df["crop"])

# =====================================================
# 🧪 BASELINE MODELS (ONLY 2 FEATURES)
# =====================================================

X_base = df[["district", "season"]]
y = df["crop"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

# ----------------------------
# 🌳 BASELINE RF
# ----------------------------
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train_b, y_train_b)

rf_pred = rf_base.predict(X_test_b)
rf_prob = rf_base.predict_proba(X_test_b)

print("\n===== BASELINE RF =====")
print("Accuracy:", accuracy_score(y_test_b, rf_pred))
print("Loss:", log_loss(y_test_b, rf_prob))

joblib.dump(rf_base, "models/rf_baseline.pkl")

# ----------------------------
# 🤖 BASELINE NN 
# ----------------------------

# NOTE: No scaling (to stay faithful to paper)

nn_base = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

nn_base.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

nn_base.fit(
    X_train_b, y_train_b,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_b, y_test_b),
    verbose=1
)

loss_b, acc_b = nn_base.evaluate(X_test_b, y_test_b, verbose=0)

print("\n===== BASELINE NN =====")
print("Accuracy:", acc_b)
print("Loss:", loss_b)

nn_base.save("models/nn_baseline.h5")

# =====================================================
# 🚀 IMPROVED MODELS (FULL FEATURES)
# =====================================================

features = [
    "district", "season", "year",
    "area", 
    "temperature", "rainfall", "humidity",
    "nitrogen", "phosphorus", "potassium",
    "organic_carbon", "ph", "micronutrient_score"
]

X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 🌳 IMPROVED RF
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)

print("\n===== IMPROVED RF =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Loss:", log_loss(y_test, rf_prob))

joblib.dump(rf_model, "models/rf_improved.pkl")

# ----------------------------
# 🤖 IMPROVED NN
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_nn.shape[1], activation='softmax')
])

nn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

nn_model.fit(
    X_train_scaled, y_train_nn,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_nn),
    verbose=1
)

loss, acc = nn_model.evaluate(X_test_scaled, y_test_nn, verbose=0)

print("\n===== IMPROVED NN =====")
print("Accuracy:", acc)
print("Loss:", loss)

nn_model.save("models/nn_improved.h5")

# ----------------------------
# SAVE ENCODERS
# ----------------------------
joblib.dump({
    "district": le_district,
    "season": le_season,
    "crop": le_crop
}, "models/encoders.pkl")

print("\n✅ FULL TRAINING COMPLETE")