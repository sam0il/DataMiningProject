import pandas as pd

# -------------------------
# 1. Load weather dataset
# -------------------------
weather_df = pd.read_csv("data/weather.csv", low_memory=False)

print("Weather loaded:")
print(weather_df.head())

# -------------------------
# 2. Remove useless column
# -------------------------
if "WindGustSpd" in weather_df.columns:
    weather_df = weather_df.drop(columns=["WindGustSpd"])

# Fill PoorWeather NaN (not used as target but keep clean)
weather_df["PoorWeather"] = weather_df["PoorWeather"].fillna(0)

# -------------------------
# 3. Fix trace values (T)
# -------------------------
weather_df["Precip"] = weather_df["Precip"].replace("T", 0.01)
weather_df["Snowfall"] = weather_df["Snowfall"].replace("T", 0.01)

# Convert to numeric
weather_df["Precip"] = pd.to_numeric(weather_df["Precip"], errors="coerce")
weather_df["Snowfall"] = pd.to_numeric(weather_df["Snowfall"], errors="coerce")

weather_df["Precip"] = weather_df["Precip"].fillna(0)
weather_df["Snowfall"] = weather_df["Snowfall"].fillna(0)

# Convert temperature columns to numeric (important for calculations)
weather_df["MaxTemp"] = pd.to_numeric(weather_df["MaxTemp"], errors="coerce")
weather_df["MinTemp"] = pd.to_numeric(weather_df["MinTemp"], errors="coerce")
weather_df["MeanTemp"] = pd.to_numeric(weather_df["MeanTemp"], errors="coerce")

print("\nWeather cleaned:")
print(weather_df.head())


# -------------------------
# 4. Load lightning dataset
# -------------------------
lightning_df = pd.read_csv("data/lightning.csv")

print("\nLightning loaded:")
print(lightning_df.head())

# Remove date column
if "date" in lightning_df.columns:
    lightning_df = lightning_df.drop(columns=["date"])

print("\nLightning cleaned:")
print(lightning_df.head())


# -------------------------
# 5. Align dataset sizes
# -------------------------
lightning_df = lightning_df.iloc[:len(weather_df)]


# -------------------------
# 6. Merge datasets
# -------------------------
merged_df = pd.concat(
    [weather_df.reset_index(drop=True),
     lightning_df.reset_index(drop=True)],
    axis=1
)

print("\nMerged dataset preview:")
print(merged_df.head())


# -------------------------
# 7. Feature engineering
# -------------------------

# Temperature range (atmospheric instability)
merged_df["TempRange"] = merged_df["MaxTemp"] - merged_df["MinTemp"]

# Storm intensity score
merged_df["storm_intensity"] = (
    merged_df["number_of_strikes"] * 0.6 + # lightning activity is a strong indicator and 0.6 weight reflects that
    merged_df["Precip"] * 0.2 +
    merged_df["TempRange"] * 0.1 +
    merged_df["MeanTemp"] * 0.1
)

print("\nFeatures added:")
print(merged_df.head())


# -------------------------
# 8. Create target variable
# -------------------------
merged_df["sprite_possible"] = (
    (merged_df["storm_intensity"] > 60) &
    (merged_df["number_of_strikes"] > 80) &
    (merged_df["Precip"] > 0.5) &
    (merged_df["MeanTemp"] > 0) &
    (merged_df["Snowfall"] == 0)
).astype(int)

print("\nDataset with target:")
print(merged_df.head())


# -------------------------
# 9. Save final dataset
# -------------------------
merged_df.to_csv("data/final_sprite_dataset.csv", index=False)
print(merged_df["sprite_possible"].value_counts())