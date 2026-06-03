import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

bitcoin = pd.read_csv("../data/bitcoin.csv", parse_dates=["date"], dayfirst=True, index_col="date")
bitcoin["value"] = pd.to_numeric(bitcoin["value"].str.replace(",", "."))

plt.figure(figsize=(12, 6))
plt.plot(bitcoin.index, bitcoin["value"], label="Bitcoin price", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Bitcoin price in time")
plt.legend()
plt.grid(True)
plt.savefig("../figures/bitcoin.png")

bitcoin["year"] = bitcoin.index.year
bitcoin["month"] = bitcoin.index.month
bitcoin["day"] = bitcoin.index.day
bitcoin["weekday"] = bitcoin.index.weekday

bitcoin["month_sin"] = np.sin(2 * np.pi * bitcoin["month"] /12)
bitcoin["month_cos"] = np.cos(2 * np.pi * bitcoin["month"] /12)
bitcoin["weekday_sin"] = np.sin(2 * np.pi * bitcoin["weekday"] / 7)
bitcoin["weekday_cos"] = np.cos(2 * np.pi * bitcoin["weekday"] / 7)

for lag in [1, 3, 7, 14]:
    bitcoin[f"value_lag_{lag}"] = bitcoin["value"].shift(lag)
    bitcoin[f"value_roll_mean_{lag}"] = bitcoin["value"].rolling(window=lag).mean().shift(lag)

bitcoin.dropna(inplace=True)

X = bitcoin.drop(columns="value")
y = bitcoin["value"]

ts_split = TimeSeriesSplit(n_splits=5)

models = {
    'LinearRegression': LinearRegression(),
    'Lasso (alpha: 0.1)': Lasso(alpha=0.1, random_state=42),
    'SVR': SVR(C=3, kernel="rbf", epsilon=0.1),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights="distance"),
    'XGBoost': XGBRegressor(max_depth=4, subsample=0.8, random_state=42),
    'RandomForest': RandomForestRegressor(max_depth=4, random_state=42)
}

results = {name: {"R²": [], "MAE": [], "RMSE": []} for name in models}
predictions = {name: [] for name in models}

# Skaler
scaler = StandardScaler()

# Trenowanie modeli i ewaluacja
for name, model in models.items():
    for i, (train_index, test_index) in enumerate(ts_split.split(X)):
        # Podział na dane treningowe i testowe
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if i == 0:
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metryki
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name]["R²"].append(r2)
        results[name]["MAE"].append(mae)
        results[name]["RMSE"].append(rmse)
        predictions[name].append((y_test.index, y_pred))
