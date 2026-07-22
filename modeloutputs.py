import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET_COLUMN = "CO2 Emissions(g/km)"

# =====================================================
# MODULE 1 : DATA COLLECTION
# =====================================================

print("\n================ MODULE 1 : DATA COLLECTION =================")

df = pd.read_csv("co2.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print(df.head())

# =====================================================
# 2. REMOVE MPG COLUMN
# =====================================================
print("\n================ MODULE 2 : DATA PREPROCESSING =================")

if "Fuel Consumption Comb (mpg)" in df.columns:
    df = df.drop("Fuel Consumption Comb (mpg)", axis=1)
    print("MPG column removed")

# =====================================================
# 3. DATA CLEANING
# =====================================================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

print("\nAfter Cleaning:")
print("Dataset Shape:", df.shape)

print("\nMissing values check:")
print(df.isnull().sum())


# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================
print("\n================ FEATURE ENGINEERING =================")

df["City_to_Hwy_Ratio"] = (
    df["Fuel Consumption City (L/100 km)"] /
    df["Fuel Consumption Hwy (L/100 km)"]
)

df["Engine_per_Cylinder"] = (
    df["Engine Size(L)"] / df["Cylinders"]
)

print("\nNew Features Added:")
print(df[["City_to_Hwy_Ratio", "Engine_per_Cylinder"]].head())


# =====================================================
# 5. ONE HOT ENCODING
# =====================================================
print("\n================ DATA TRANSFORMATION =================")

df = pd.get_dummies(df, drop_first=True)

print("\nAfter One-Hot Encoding")
print("Dataset Shape:", df.shape)


# =====================================================
# 6. SPLIT FEATURES & TARGET
# =====================================================
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

feature_names = X.columns.tolist()

print("\nTotal Features Used:", len(feature_names))


# =====================================================
# 7. TRAIN TEST SPLIT
# =====================================================
print("\n================ DATA SPLITTING =================")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)


# =====================================================
# 8. FEATURE SCALING
# =====================================================
print("\n================ FEATURE SCALING =================")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Standard Scaling Applied Successfully")

# =====================================================
# MODULE 3 : MODEL TRAINING
# =====================================================

print("\n================ MODULE 3 : MODEL TRAINING =================")

lr_model = LinearRegression()

rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=22,
    min_samples_split=3,
    random_state=42
)

# DEFAULT SVR (Before tuning)
svr_model = SVR()

# Train models
lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)
svr_model.fit(X_train_scaled, y_train)

# Stacking model
stack_model = StackingRegressor(
    estimators=[
        ("lr", lr_model),
        ("rf", rf_model),
        ("svr", svr_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stack_model.fit(X_train_scaled, y_train)

print("Models trained successfully")

# =====================================================
# MODULE 4 : MODEL VALIDATION (BEFORE TUNING)
# =====================================================

print("\n================ MODULE 4 : MODEL VALIDATION (BEFORE TUNING) =================")

def evaluate(name, y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("R2 Score :", round(r2,4))
    print("MAE :", round(mae,2))
    print("RMSE :", round(rmse,2))

lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)
svr_pred = svr_model.predict(X_test_scaled)
stack_pred = stack_model.predict(X_test_scaled)

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("SVR (Before Tuning)", y_test, svr_pred)
evaluate("Stacking Ensemble", y_test, stack_pred)

# =====================================================
# MODULE 5 : FINE TUNING (SVR)
# =====================================================

print("\n================ MODULE 5 : FINE TUNING =================")

svr_tuned = SVR(
    kernel="rbf",
    C=120,
    gamma=0.08,
    epsilon=0.1
)

svr_tuned.fit(X_train_scaled, y_train)

print("SVR Fine Tuning Completed")

# =====================================================
# MODULE 6 : MODEL VALIDATION AFTER TUNING
# =====================================================

print("\n================ MODULE 6 : MODEL VALIDATION AFTER TUNING =================")

svr_pred_tuned = svr_tuned.predict(X_test_scaled)

# New stacking model with tuned SVR
stack_model_tuned = StackingRegressor(
    estimators=[
        ("lr", lr_model),
        ("rf", rf_model),
        ("svr", svr_tuned)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stack_model_tuned.fit(X_train_scaled, y_train)

stack_pred_tuned = stack_model_tuned.predict(X_test_scaled)

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("SVR (After Tuning)", y_test, svr_pred_tuned)
evaluate("Stacking Ensemble (After Tuning)", y_test, stack_pred_tuned)

# =====================================================
# RESULT VISUALIZATION
# =====================================================

plt.figure(figsize=(6,6))

plt.scatter(y_test, stack_pred_tuned)

min_val = min(min(y_test), min(stack_pred_tuned))
max_val = max(max(y_test), max(stack_pred_tuned))

plt.plot([min_val,max_val],[min_val,max_val])

plt.xlabel("Actual CO2")
plt.ylabel("Predicted CO2")
plt.title("Actual vs Predicted CO2 Emissions")

plt.show()

# =====================================================
# MODULE 7 : MODEL DEPLOYMENT
# =====================================================

print("\n================ MODULE 7 : MODEL DEPLOYMENT =================")

os.makedirs("model", exist_ok=True)

joblib.dump(lr_model,"model/lr_model.pkl")
joblib.dump(rf_model,"model/rf_model.pkl")
joblib.dump(svr_tuned,"model/svr_model.pkl")
joblib.dump(stack_model_tuned,"model/stack_model.pkl")
joblib.dump(scaler,"model/scaler.pkl")
joblib.dump(feature_names,"model/features.pkl")

print("Models saved successfully")

print("\nPROJECT COMPLETED SUCCESSFULLY")