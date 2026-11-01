import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

auto = pd.read_csv("auto_mpg.csv")
print("Shape:", auto.shape, "| missing horsepower:", auto["horsepower"].isnull().sum())
auto = auto.dropna(subset=["horsepower"])
print("Shape after dropping missing:", auto.shape)
X = auto[["horsepower"]].values; y = auto["mpg"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 5, 8, 10]
rows, models = [], {}
for d in degrees:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())
    m.fit(X_train, y_train); models[d] = m
    rows.append([d, mean_squared_error(y_train, m.predict(X_train)),
                 mean_squared_error(y_test, m.predict(X_test))])
res = pd.DataFrame(rows, columns=["Degree", "Train MSE", "Test MSE"])
print("\n", res.round(3).to_string(index=False))
print("\nOptimal degree d* (min test MSE):", res.loc[res["Test MSE"].idxmin(), "Degree"])

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
ax[0].scatter(X, y, s=6, color="#bbbbbb")
for d, c in zip([1, 2, 10], ["#2a78d6", "#1baf7a", "#eb6834"]):
    ax[0].plot(xs, models[d].predict(xs), color=c, label=f"degree {d}")
ax[0].set_ylim(5, 50); ax[0].set_xlabel("Horsepower"); ax[0].set_ylabel("MPG"); ax[0].legend()
ax[0].set_title("Fitted polynomial curves")
ax[1].plot(res["Degree"], res["Train MSE"], "o-", label="Train MSE")
ax[1].plot(res["Degree"], res["Test MSE"], "o-", label="Test MSE")
ax[1].set_xlabel("Polynomial degree"); ax[1].set_ylabel("MSE"); ax[1].legend()
ax[1].set_title("Model complexity curve"); plt.tight_layout()
