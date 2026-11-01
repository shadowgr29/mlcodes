import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = sns.load_dataset("titanic")

df = df.rename(columns={
    "survived": "survived",
    "pclass": "pclass",
    "sex": "sex",
    "age": "age",
    "sibsp": "sibsp",
    "parch": "parch",
    "fare": "fare",
    "embarked": "embarked"
})

print("Shape:", df.shape)

data = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

print(data.head(), "\n")
print("Missing values:\n", data.isnull().sum()[data.isnull().sum() > 0], "\n")

age_before = data["age"].copy()

print("Median Age:", data["age"].median())
data["age"] = data["age"].fillna(data["age"].median())

print("Mode of Embarked:", data["embarked"].mode()[0])
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])

print("Sex encoding:", {c: int(v) for c, v in zip(le.classes_, le.transform(le.classes_))})

data = pd.get_dummies(data, columns=["embarked"], dtype=int)

num_cols = ["age", "sibsp", "parch", "fare"]
data[num_cols] = StandardScaler().fit_transform(data[num_cols])

print("\nMissing values after cleaning:", int(data.isnull().sum().sum()))
print("Final shape:", data.shape)
print("Mean of scaled cols:", data[num_cols].mean().round(4).to_dict())
print("Std of scaled cols:", data[num_cols].std(ddof=0).round(4).to_dict())
print("\nClean, model-ready dataset:\n", data.head().round(4))

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].hist(age_before.dropna(), bins=30, edgecolor="white")
ax[0].set_title("Age before imputation")
ax[1].hist(data["age"], bins=30, edgecolor="white")
ax[1].set_title("Age after median imputation and scaling")
plt.tight_layout()
plt.show()