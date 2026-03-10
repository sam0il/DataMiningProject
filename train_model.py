import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# ----------------------------------------------------------------------------------------------------
# 1. Load dataset
df = pd.read_csv("data/final_sprite_dataset.csv")

#print("Dataset loaded:")
#print(df.head())


# ----------------------------------------------------------------------------------------------------
# 2. Features (X)
X = df[[
    "Precip",
    "MaxTemp",
    "MinTemp",
    "MeanTemp",
    "Snowfall",
    "number_of_strikes",
    "TempRange"
]]

# ----------------------------------------------------------------------------------------------------
# 3. Target (y)
y = df["sprite_possible"]


# ----------------------------------------------------------------------------------------------------
# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#print("\nTraining samples:", len(X_train))
#print("Testing samples:", len(X_test))


# ----------------------------------------------------------------------------------------------------
# 5. Feature Scaling
scaler = StandardScaler()
# Fit on training data and transform it
X_train = scaler.fit_transform(X_train)
# Only transform test data
X_test = scaler.transform(X_test)
#print("\nFeatures scaled.")


# ----------------------------------------------------------------------------------------------------
# 6. Train Logistic Regression model
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
print("\nModel training complete.")

# ----------------------------------------------------------------------------------------------------
# 7. Make predictions
#y_pred = model.predict(X_test)
#print("\nPredictions made.")


# Threshold testing
threshold = 0.40

y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > threshold).astype(int)

print("\nUsing threshold:", threshold)


# ----------------------------------------------------------------------------------------------------
# 8. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)


# ----------------------------------------------------------------------------------------------------
# 9. Confusion Matrix
                                        #                  0              1
                                        # Actual 0   TrueNegative   FalsePositive
                                        # Actual 1   FalseNegative  TruePositive
cm = confusion_matrix(y_test, y_pred) 
print("\nConfusion Matrix:")
print(cm)


# ----------------------------------------------------------------------------------------------------
# 10. Precision / Recall / F1
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ----------------------------------------------------------------------------------------------------
# 11. Feature importance
feature_names = X.columns

for feature, coef in zip(feature_names, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


# ----------------------------------------------------------------------------------------------------
    # ROC Curve
y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()