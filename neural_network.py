import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance


df = pd.read_csv("data/final_sprite_dataset_withtarget.csv")

X = df[[ 
   #"Precip",
   "MaxTemp",
   "MinTemp",
   "MeanTemp",
   "Snowfall",
  # "Thunder", #also huge factor even bigger than number_strikes which was weird. for model.
   "Sleet",
   "Hail",
   "Dust/Sand",
   "Smoke/Haze",
   "BlowingSnow",
   "Rain",
   "Snow",
   "Glaze",
   "Fog",
   #"number_of_strikes",
   #"Discrimination" #from 16 to 65 false alarms. when we remove it it doesn't stop at itteration 32. for model.
]]

Y = df["sprite_possibility"]


#print("Total dataset:")
#print(Y.value_counts())
#print(f"Total sprites: {Y.sum()}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

#print("Training set:")
#print(Y_train.value_counts())

#print("\nTest set:")
#print(Y_test.value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#print("\nScaling done.")
#print(f"X_train shape: {X_train_scaled.shape}")
#print(f"X_test shape:  {X_test_scaled.shape}")


model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  #1 output as default by library.
    activation="relu",#|'relu: if - then 0 else keep', 'identity: turns into linear function', 'logistic: from 0 to 1', 'tanh from -1 to 1'|
    max_iter=500,
    solver="adam", #'lbfgs', 'sgd', 'adam'
    alpha=0.0001, #regularization, adding a penalty term to the cost function so it doesn't overfit.
    batch_size="auto",
    learning_rate="constant", #|'constant', 'invscaling', 'adaptive'| #need to explain.
    learning_rate_init=0.001,#need to explain.
    shuffle=True,
    tol=0.0001, #how small the improvement needs to be before it considers itself "done"
    verbose=False, #if True it prints the loss every iteration
    warm_start=False, #if true continues not new start.
    beta_1=0.9, beta_2=0.999, epsilon=1e-8,   #constants for adam             
    random_state=42
    # removed early_stopping — it was stopping too early due to class imbalance
    # the balanced sample_weight below handles the imbalance instead and it still stops at 32 idk why.
    #------------------------------------------------------------------------------------------------------
    # momentum=0.9 used in sgd
    #nesterovs_momentum=True also in sgd
    #validation_fraction=0.1 used in early_stopping
    #n_iter_no_change=10 used in early_stopping
    #max_fun=15000 used in lbfgs

)

model2 = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    solver="sgd",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    shuffle=True ,
    tol=0.0001,
    verbose=False,
    warm_start=False,

    nesterovs_momentum=True,
    momentum=0.9,
    #---
    #early_stopping=True,
    #validation_fraction=0.1,
    #n_iter_no_change=10,
    #---
    random_state=42

)

model3 = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=max(500, 15000), #lbfgs needs more iterations to converge.
    solver="lbfgs",
    alpha=0.0001,
    #batch_size="auto", ignored by lbfgs
    #learning_rate="constant", ignored by lbfgs
    #learning_rate_init=0.001, ignored by lbfgs
    #shuffle=True , ignored by lbfgs
    tol=0.0001,
    verbose=False,
    warm_start=False,

    max_fun=15000,
    #---
    #early_stopping=True, ignored by lbfgs
    #validation_fraction=0.1, ignored by lbfgs
    #n_iter_no_change=10, ignored by lbfgs
    #---
    random_state=42

)


sample_weights = compute_sample_weight(class_weight="balanced", y=Y_train)
model.fit(X_train_scaled, Y_train, sample_weight=sample_weights)
model2.fit(X_train_scaled, Y_train, sample_weight=sample_weights)
model3.fit(X_train_scaled, Y_train, sample_weight=sample_weights)

print("Models trained.")
print(f"Stopped at iteration (Adam): {model.n_iter_}")
print(f"Stopped at iteration (SGD): {model2.n_iter_}")
print(f"Stopped at iteration (LBFGS): {model3.n_iter_}")


Y_pred = model.predict(X_test_scaled)
Y_pred2 = model2.predict(X_test_scaled)
Y_pred3 = model3.predict(X_test_scaled)


#Add model2 and model3
print("Accuracy (Adam):", accuracy_score(Y_test, Y_pred))
print("Accuracy (SGD):", accuracy_score(Y_test, Y_pred2))
print("Accuracy (LBFGS):", accuracy_score(Y_test, Y_pred3))

print("\nConfusion Matrix (Adam):")
print(confusion_matrix(Y_test, Y_pred))
print("\nConfusion Matrix (SGD):")
print(confusion_matrix(Y_test, Y_pred2))
print("\nConfusion Matrix (LBFGS):")
print(confusion_matrix(Y_test, Y_pred3))

print("\nClassification Report (Adam):")
print(classification_report(Y_test, Y_pred, target_names=["Non-Favorable", "Sprite-Favorable"]))
print("\nClassification Report (SGD):")
print(classification_report(Y_test, Y_pred2, target_names=["Non-Favorable", "Sprite-Favorable"]))
print("\nClassification Report (LBFGS):")
print(classification_report(Y_test, Y_pred3, target_names=["Non-Favorable", "Sprite-Favorable"]))




for mdl, name in [(model, "Adam"), (model2, "SGD"), (model3, "LBFGS")]:
    result = permutation_importance(
        mdl, X_test_scaled, Y_test,
        n_repeats=10,
        random_state=42
    )
    importance_df = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=False)

    print(f"\nFeature Importance ({name}):")
    print(importance_df.to_string(index=False))