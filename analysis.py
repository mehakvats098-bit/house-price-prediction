import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/house_prices.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Information ---")
df.info()


# ==========================================
# BASIC STATISTICAL ANALYSIS
# ==========================================

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Average House Price ---")
print(df["Price_Lakh"].mean())

print("\n--- Minimum House Price ---")
print(df["Price_Lakh"].min())

print("\n--- Maximum House Price ---")
print(df["Price_Lakh"].max())

print("\n--- Average House Area ---")
print(df["Area_sqft"].mean())


print("\n--- Average Price by Location ---")

avg_price_location = (
    df.groupby("Location")["Price_Lakh"]
    .mean()
)

print(avg_price_location)


# ==========================================
# PRICE ANALYSIS BY FEATURES
# ==========================================

print("\n--- Average Price by Bedrooms ---")

price_by_bedrooms = (
    df.groupby("Bedrooms")["Price_Lakh"]
    .mean()
)

print(price_by_bedrooms)


print("\n--- Average Price by Parking ---")

price_by_parking = (
    df.groupby("Parking")["Price_Lakh"]
    .mean()
)

print(price_by_parking)


print("\n--- Average Price by Location ---")

price_by_location = (
    df.groupby("Location")["Price_Lakh"]
    .mean()
)

print(price_by_location)


print("\n--- Average Price by Bathrooms ---")

price_by_bathrooms = (
    df.groupby("Bathrooms")["Price_Lakh"]
    .mean()
)

print(price_by_bathrooms)


# ==========================================
# CREATE CHARTS
# ==========================================

os.makedirs("charts", exist_ok=True)


# ------------------------------------------
# 1. House Price Distribution
# ------------------------------------------

plt.figure(figsize=(7, 5))

df["Price_Lakh"].plot(
    kind="hist",
    bins=10
)

plt.title("House Price Distribution")
plt.xlabel("Price (Lakh)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.savefig("charts/price_distribution.png")
plt.close()


# ------------------------------------------
# 2. Area vs Price
# ------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Area_sqft"],
    df["Price_Lakh"]
)

plt.title("House Area vs Price")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakh)")

plt.tight_layout()
plt.savefig("charts/area_vs_price.png")
plt.close()


# ------------------------------------------
# 3. Average Price by Location
# ------------------------------------------

price_by_location.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average House Price by Location")
plt.xlabel("Location")
plt.ylabel("Average Price (Lakh)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("charts/price_by_location.png")
plt.close()


# ------------------------------------------
# 4. Average Price by Bedrooms
# ------------------------------------------

price_by_bedrooms.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average House Price by Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Average Price (Lakh)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("charts/price_by_bedrooms.png")
plt.close()


print("\n--- Charts Created Successfully ---")


# ==========================================
# PREPARE DATA FOR MACHINE LEARNING
# ==========================================

# Features
X = df.drop(
    columns=["House_ID", "Price_Lakh"]
)

# Target
y = df["Price_Lakh"]

print("\n--- Features ---")
print(X.head())

print("\n--- Target ---")
print(y.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

X_encoded = pd.get_dummies(
    X,
    columns=[
        "Location",
        "Parking"
    ],
    drop_first=True
)

print("\n--- Encoded Features ---")
print(X_encoded.head())

print("\nEncoded feature shape:", X_encoded.shape)


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Train-Test Split ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ==========================================
# TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\n--- Model Training ---")
print("Linear Regression model trained successfully!")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

print("\n--- Predictions ---")
print(y_pred)

print("\n--- Actual Prices ---")
print(y_test.values)


# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n--- Model Evaluation ---")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


# ==========================================
# ACTUAL VS PREDICTED PRICES
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price (Lakh)")
plt.ylabel("Predicted Price (Lakh)")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.savefig("charts/actual_vs_predicted.png")
plt.close()

print("\nActual vs predicted chart saved!")


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/house_price_model.pkl"
)

print("\nModel saved successfully!")


# ==========================================
# SAVE FEATURE COLUMNS
# ==========================================

feature_columns = X_encoded.columns.tolist()

joblib.dump(
    feature_columns,
    "model/feature_columns.pkl"
)

print("\nFeature columns saved successfully!")


# ==========================================
# SAVE MODEL EVALUATION REPORT
# ==========================================

with open(
    "model/evaluation_report.txt",
    "w"
) as file:

    file.write(
        "House Price Prediction - Model Evaluation\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"MAE: {mae:.4f}\n"
    )

    file.write(
        f"MSE: {mse:.4f}\n"
    )

    file.write(
        f"RMSE: {rmse:.4f}\n"
    )

    file.write(
        f"R2 Score: {r2:.4f}\n"
    )


print("\nEvaluation report saved successfully!")

print("\n==========================================")
print("PROJECT 9 ANALYSIS COMPLETED SUCCESSFULLY")
print("==========================================")