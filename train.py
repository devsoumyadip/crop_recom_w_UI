# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, log_loss

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# # ----------------------------
# # LOAD DATA
# # ----------------------------
# df = pd.read_csv("./data/west_bengal_cleaned_crop.csv")

# # Clean
# df.columns = df.columns.str.lower().str.strip()
# for col in ["district", "season", "crop"]:
#     df[col] = df[col].str.lower().str.strip()

# # ----------------------------
# # CREATE YIELD SCORE (CRITICAL STEP FROM PAPER)
# # ----------------------------
# df["yield_score"] = df["yield"] * df["production"]

# # ----------------------------
# # ENCODING
# # ----------------------------
# le_district = LabelEncoder()
# le_season = LabelEncoder()
# le_crop = LabelEncoder()

# df["district"] = le_district.fit_transform(df["district"])
# df["season"] = le_season.fit_transform(df["season"])
# df["crop"] = le_crop.fit_transform(df["crop"])

# # ----------------------------
# # FEATURES (ONLY 2 AS PER PAPER)
# # ----------------------------
# X = df[["district", "season"]]
# y = df["crop"]

# # ----------------------------
# # SPLIT (80-20)
# # ----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # =====================================================
# # 🌳 RANDOM FOREST (EXACT PAPER)
# # =====================================================
# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42
# )

# rf_model.fit(X_train, y_train)

# rf_pred = rf_model.predict(X_test)
# rf_prob = rf_model.predict_proba(X_test)

# rf_acc = accuracy_score(y_test, rf_pred)
# rf_loss = log_loss(y_test, rf_prob, labels=np.unique(y_train))

# print("\n==============================")
# print("Method: Random Forest")
# print("Accuracy:", rf_acc)
# print("Loss:", rf_loss)
# print("Type of Loss: Log Loss")
# print("==============================")

# # =====================================================
# # 🤖 FEEDFORWARD NN (EXACT PAPER)
# # =====================================================

# # NO scaling used in paper (important detail)

# # Model
# nn_model = Sequential([
#     Dense(64, activation='relu', input_shape=(2,)),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(len(np.unique(y)), activation='softmax')
# ])

# nn_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train (50 epochs exactly as paper)
# nn_model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     verbose=1
# )

# # Evaluate
# nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)

# print("\n==============================")
# print("Method: Feedforward Neural Network")
# print("Accuracy:", nn_acc)
# print("Loss:", nn_loss)
# print("Type of Loss: Sparse Categorical Crossentropy")
# print("==============================")

# # ----------------------------
# # SAVE MODELS
# # ----------------------------
# joblib.dump(rf_model, "models/rf_model.pkl")

# nn_model.save("models/nn_model.h5")

# joblib.dump({
#     "district": le_district,
#     "season": le_season,
#     "crop": le_crop
# }, "models/encoders.pkl")

# print("\n✅ Exact paper training complete")



############
# proposed #
###########




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
df = pd.read_csv("./data/west_bengal_cleaned_crop.csv")

df.columns = df.columns.str.lower().str.strip()

# Clean text
for col in ["district", "season", "crop"]:
    df[col] = df[col].str.lower().str.strip()

# ----------------------------
# 🔥 FILTER TOP CROPS
# ----------------------------
top_crops = df["crop"].value_counts().head(6).index
df = df[df["crop"].isin(top_crops)]

# ----------------------------
# 🔥 FEATURE ENGINEERING
# ----------------------------
df["productivity"] = df["production"] / (df["area"] + 1)

# ----------------------------
# 🔥 BALANCE DATASET
# ----------------------------
df = df.groupby("crop").apply(lambda x: x.sample(min(len(x), 500), random_state=42))
df = df.reset_index(drop=True)

# ----------------------------
# ENCODING
# ----------------------------
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df["district"] = le_district.fit_transform(df["district"])
df["season"] = le_season.fit_transform(df["season"])
df["crop"] = le_crop.fit_transform(df["crop"])

# ----------------------------
# FEATURES (IMPROVED BASELINE)
# ----------------------------
X = df[[
    "district", "season", "year",
    "area", "production", "productivity"
]]

y = df["crop"]

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 🌳 RANDOM FOREST (TUNED)
# =====================================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_loss = log_loss(y_test, rf_prob)

print("\n==============================")
print("Method: Random Forest (Improved)")
print("Accuracy:", rf_acc)
print("Loss:", rf_loss)
print("Type of Loss: Log Loss")
print("==============================")

joblib.dump(rf_model, "models/rf_model_improved.pkl")

# =====================================================
# 🤖 NEURAL NETWORK (IMPROVED)
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler_improved.pkl")

y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

# Improved architecture
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_nn.shape[1], activation='softmax')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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

# Evaluate
nn_loss, nn_acc = nn_model.evaluate(X_test_scaled, y_test_nn, verbose=0)

print("\n==============================")
print("Method: Feedforward Neural Network (Improved)")
print("Accuracy:", nn_acc)
print("Loss:", nn_loss)
print("Type of Loss: Categorical Crossentropy")
print("==============================")

nn_model.save("models/nn_model_improved.h5")

# ----------------------------
# SAVE ENCODERS
# ----------------------------
joblib.dump({
    "district": le_district,
    "season": le_season,
    "crop": le_crop
}, "models/encoders_improved.pkl")

print("\n✅ Improved baseline training complete")