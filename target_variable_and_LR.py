import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_csv("data/final_sprite_cleaned_dataset.csv")

final_df["sprite_possibility"] = (
    (final_df["number_of_strikes"] > 80) &
    (final_df["Discrimination"] == 1) &
    (final_df["Thunder"] == 1) &
    (final_df["Precip"] > 0.5)
).astype(int)  # convert True/False to 1/0

#final_df.to_csv("data/final_sprite_dataset_with_target.csv", index=False)

#test_df = pd.read_csv("data/final_sprite_dataset_with_target.csv")
#print(test_df["sprite_possibility"].unique())
#print(test_df["sprite_possibility"].value_counts())
#----------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("data/final_sprite_dataset_withtarget.csv")

X = df[[ 
   #"Precip",
   "MaxTemp",
   "MinTemp",
   "MeanTemp",
   "Snowfall",
   #"Thunder", #also huge factor even bigger than number_strikes which was weird.
   "Sleet",
   "Hail",
   "Dust/Sand",
   "Smoke/Haze",
   "BlowingSnow",
   "Rain",
   "Snow",
   "Glaze",
   "Fog"
  # "number_of_strikes",
  # "Discrimination" #huge huge factor for the model from 23 false alarms to 179 when removed.
]]

Y = df["sprite_possibility"]

# Check before splitting
print("Total dataset:")
print(Y.value_counts())
print(f"Total sprites: {Y.sum()}")



#split the data into training and testing sets 80% train 20% test.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# Check after splitting
print("\nTraining set:")
print(Y_train.value_counts())
print(f"Training sprites: {Y_train.sum()}")

print("\nTest set:")
print(Y_test.value_counts())
print(f"Test sprites: {Y_test.sum()}")


#scale the features from -2 to 2 using StandardScaler. So big values don't dominate the model.
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
model_balanced.fit(X_train, Y_train)

print("Model trained.")

#predicting
Y_pred = model.predict(X_test)
Y_pred_balanced = model_balanced.predict(X_test)


#charts and evaluation 
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Accuracy (Balanced):", accuracy_score(Y_test, Y_pred_balanced))

print("\nConfusion Matrix:")                     
print(confusion_matrix(Y_test, Y_pred))             #                  0              1
print("\nConfusion Matrix (Balanced):")             # Actual 0   TrueNegative   FalsePositive
print(confusion_matrix(Y_test, Y_pred_balanced))    # Actual 1   FalseNegative  TruePositive                                         
                                                   


print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=["Non-Favorable", "Sprite-Favorable"], zero_division=0));
print("\nClassification Report (Balanced):")
print(classification_report(Y_test, Y_pred_balanced, target_names=["Non-Favorable", "Sprite-Favorable"]))

#this is from claude...
feature_names = X.columns.tolist()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient (Default)": model.coef_[0],
    "Coefficient (Balanced)": model_balanced.coef_[0]
}).sort_values("Coefficient (Default)", ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients):")
print(importance_df.to_string(index=False))