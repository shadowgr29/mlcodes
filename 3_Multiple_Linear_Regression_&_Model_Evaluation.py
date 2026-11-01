import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

raw = pd.read_csv("housing.csv")
df = pd.DataFrame({
    "MedInc": raw["median_income"], "HouseAge": raw["housing_median_age"],
    "AveRooms": raw["total_rooms"] / raw["households"],
    "AveBedrms": raw["total_bedrooms"] / raw["households"],
    "Population": raw["population"], "AveOccup": raw["population"] / raw["households"],
    "Latitude": raw["latitude"], "Longitude": raw["longitude"],
    "MedHouseVal": raw["median_house_value"] / 100000})
print("Shape:", df.shape)
print(df.head().round(3), "\n")
print("Missing values:", df.isnull().sum().to_dict())
df["AveBedrms"] = df["AveBedrms"].fillna(df["AveBedrms"].median())

X = df.drop(columns="MedHouseVal"); y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMSE : %.4f" % mse)
print("RMSE : %.4f" % np.sqrt(mse))
print("R2   : %.4f" % r2_score(y_test, y_pred))
print("\nIntercept: %.4f" % model.intercept_)
print("Coefficients:\n", pd.Series(model.coef_, index=X.columns).round(4).to_string())

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].scatter(y_test, y_pred, s=4, alpha=0.3, color="#2a78d6")
ax[0].plot([0, 5], [0, 5], color="#eb6834"); ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
ax[0].set_title("Actual vs Predicted")
ax[1].hist(y_test - y_pred, bins=50, color="#1baf7a", edgecolor="white"); ax[1].set_title("Residuals")
plt.tight_layout()
