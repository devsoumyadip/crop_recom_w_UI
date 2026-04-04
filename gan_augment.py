import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("./data/finalData.csv")

df.columns = df.columns.str.lower().str.strip()
df = df.drop(columns=["unnamed: 0"], errors="ignore")

for col in ["district", "season", "crop"]:
    df[col] = df[col].astype(str)

# ----------------------------
# CREATE METADATA
# ----------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# ----------------------------
# TRAIN CTGAN
# ----------------------------
ctgan = CTGANSynthesizer(metadata, epochs=100)

ctgan.fit(df)

# ----------------------------
# GENERATE SYNTHETIC DATA
# ----------------------------
synthetic_data = ctgan.sample(num_rows=len(df))

# ----------------------------
# SAVE
# ----------------------------
synthetic_data.to_csv("./data/synthetic_data.csv", index=False)

print("✅ Synthetic data generated successfully")