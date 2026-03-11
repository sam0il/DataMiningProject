import pandas as pd

weather_df = pd.read_csv("data/weather.csv", low_memory=False)
weather_df = weather_df.rename(columns={"PoorWeather": "PoorWeather_TSHDSBRSGF"}) #Thunder; Sleet; Hail; Dust or Sand; Smoke or Haze; Blowing Snow; Rain; Snow; Glaze; Fog;
weather_df["PoorWeather_TSHDSBRSGF"] = weather_df["PoorWeather_TSHDSBRSGF"].fillna(0) #Fills in NaN values with 0.
weather_df["PoorWeather_TSHDSBRSGF"] = weather_df["PoorWeather_TSHDSBRSGF"].astype(str).str.replace(" ", "0", regex=False)
weather_df["PoorWeather_TSHDSBRSGF"] = weather_df["PoorWeather_TSHDSBRSGF"].str.ljust(10, "0")

TSHDSBRSGF_cols = [
    "Thunder",      
    "Sleet",        
    "Hail",         
    "Dust/Sand",         
    "Smoke/Haze",    
    "BlowingSnow",  
    "Rain",         
    "Snow",         
    "Glaze",     
    "Fog"           
]

for i, col in enumerate(TSHDSBRSGF_cols):
    weather_df[col] = weather_df["PoorWeather_TSHDSBRSGF"].str[i].astype(int)

#print(weather_df[TSHDSBRSGF_cols].head())
#print(weather_df[TSHDSBRSGF_cols].head(20))

weather_df["Snowfall"] = weather_df["Snowfall"].replace("#VALUE!", pd.NA)
weather_df["Snowfall"] = pd.to_numeric(weather_df["Snowfall"], errors="coerce")
weather_df["Snowfall"] = weather_df["Snowfall"].fillna(0)
#print(weather_df["Snowfall"].value_counts(dropna=False))

weather_df["Precip"] = weather_df["Precip"].replace("T", 0.1)
weather_df["Precip"] = pd.to_numeric(weather_df["Precip"], errors="coerce")
#print(weather_df["Precip"].value_counts(dropna=False))

weather_df.drop(columns=["PoorWeather_TSHDSBRSGF"], inplace=True) #Inplace=True means that the original DataFrame will be modified and the column will be removed from it
#print(weather_df.head(20));
#weather_df.to_csv("data/weather_cleaned.csv", index=False)

weather_cleaned_df = pd.read_csv("data/weather_cleaned.csv")
#print(weather_cleaned_df.head(20));
#print(weather_cleaned_df.nunique())

lightning_df = pd.read_csv("data/lightning.csv")
lightning_df.drop(columns=["date"], inplace=True)
#print(lightning_df.head(20));
#print(lightning_df.nunique())
#lightning_df.to_csv("data/lightning_cleaned.csv", index=False)

positive_negative_df = pd.read_csv("data/positive_negative_strikes.csv")
print(weather_df.shape)
print(lightning_df.shape)
print(positive_negative_df.shape)

#Will not work since the datasets have diffrent numbers of rows. We will need to align them first before merging! (will use the smallest dataset as the reference for alignment)
#final_df = pd.concat([weather_df, lightning_df, positive_negative_df], axis=1) #axis=1 means that the concatenation will be done horizontally.
#print(final_df.head(20));