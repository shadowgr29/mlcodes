# EXPERIMENT NO. 01 – Basic Feature Engineering Techniques 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("titanic.csv")
print("Shape:", df.shape)
print(df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].head(), "\n")

print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0], "\n")
data = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
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
print("Std of scaled cols :", data[num_cols].std(ddof=0).round(4).to_dict())
print("\nClean, model-ready dataset:\n", data.head().round(4))

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].hist(age_before.dropna(), bins=30, color="#2a78d6", edgecolor="white"); ax[0].set_title("Age before imputation")
ax[1].hist(data["age"], bins=30, color="#eb6834", edgecolor="white"); ax[1].set_title("Age after median imputation + scaling")
plt.tight_layout()

# EXPERIMENT NO. 02 – Dimensionality Reduction using PCA and LDA 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
print("Dataset shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

def evaluate(Xtr, Xte, name):
    t0 = time.perf_counter()
    for _ in range(50):                     
        clf = LogisticRegression(max_iter=5000).fit(Xtr, y_train)
    t = (time.perf_counter() - t0) / 50 * 1000
    acc = accuracy_score(y_test, clf.predict(Xte))
    return [name, Xtr.shape[1], round(acc, 4), round(t, 3)]

r1 = evaluate(X_train, X_test, "Original")

pca = PCA(n_components=10)
Xtr_p = pca.fit_transform(X_train); Xte_p = pca.transform(X_test)
print("Variance retained by 10 PCs: %.2f%%" % (pca.explained_variance_ratio_.sum() * 100))
r2 = evaluate(Xtr_p, Xte_p, "PCA")

lda = LDA(n_components=1)
Xtr_l = lda.fit_transform(X_train, y_train); Xte_l = lda.transform(X_test)
r3 = evaluate(Xtr_l, Xte_l, "LDA")

res = pd.DataFrame([r1, r2, r3], columns=["Case", "Features", "Accuracy", "Train time (ms)"])
print("\n", res.to_string(index=False))

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
for c, col, lab in [(0, "#eb6834", "malignant"), (1, "#2a78d6", "benign")]:
    ax[0].scatter(Xtr_p[y_train == c, 0], Xtr_p[y_train == c, 1], s=10, color=col, label=lab)
    ax[1].hist(Xtr_l[y_train == c, 0], bins=30, color=col, alpha=0.75, label=lab)
ax[0].set_title("PCA: PC1 vs PC2"); ax[1].set_title("LDA: single component")
ax[0].legend(); ax[1].legend(); plt.tight_layout()

# EXPERIMENT NO. 03 – Multiple Linear Regression 
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

# EXPERIMENT NO. 04 – Polynomial Regression and Bias–Variance Trade-off 
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

# EXPERIMENT NO. 05 – Ridge and Lasso Regularization Techniques 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = load_diabetes(as_frame=True).frame
print("Shape:", df.shape, "| Missing values:", int(df.isnull().sum().sum()))
X = df.drop(columns="target"); y = df["target"]

corr = X.corr().abs().where(np.triu(np.ones((10, 10)), 1).astype(bool)).stack()
print("Most correlated pairs:\n", corr.sort_values(ascending=False).head(3).round(3).to_string())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

models = {"OLS": LinearRegression(), "Ridge": Ridge(alpha=53.37), "Lasso": Lasso(alpha=0.1485)}
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    p_tr, p_te = m.predict(X_train), m.predict(X_test)
    rows.append([name, mean_squared_error(y_train, p_tr), mean_squared_error(y_test, p_te),
                 np.sqrt(mean_squared_error(y_test, p_te)), r2_score(y_train, p_tr), r2_score(y_test, p_te)])
print("\n", pd.DataFrame(rows, columns=["Model", "Train MSE", "Test MSE", "Test RMSE", "Train R2", "Test R2"])
      .round(3).to_string(index=False))

coefs = pd.DataFrame({n: m.coef_ for n, m in models.items()}, index=X.columns)
print("\nCoefficients:\n", coefs.round(2).to_string())
print("\nFeatures removed by Lasso:", list(coefs.index[coefs["Lasso"] == 0]))

alphas = np.logspace(-2, 2, 100)
path = np.array([Lasso(alpha=a, max_iter=50000).fit(X_train, y_train).coef_ for a in alphas])
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
coefs.plot(kind="bar", ax=ax[0], width=0.8); ax[0].set_title("OLS vs Ridge vs Lasso coefficients")
ax[1].plot(alphas, path); ax[1].set_xscale("log"); ax[1].set_xlabel("alpha")
ax[1].set_title("Lasso path (coefficients shrink to 0)"); plt.tight_layout()

# EXPERIMENT NO. 06 – Logistic Regression for Binary Classification 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, roc_auc_score)

X, y = load_breast_cancer(return_X_y=True)          # 0 = malignant, 1 = benign
print("Shape:", X.shape, "| Class counts (malignant, benign):", np.bincount(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

clf = LogisticRegression(max_iter=5000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
p_mal = clf.predict_proba(X_test)[:, 0]

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (rows = actual, cols = predicted; malignant = positive):\n", cm)
print("\nAccuracy : %.4f" % accuracy_score(y_test, y_pred))
print("Precision : %.4f" % precision_score(y_test, y_pred, pos_label=0))
print("Recall    : %.4f" % recall_score(y_test, y_pred, pos_label=0))
print("F1-score : %.4f" % f1_score(y_test, y_pred, pos_label=0))
print("ROC-AUC   : %.4f" % roc_auc_score(y_test == 0, p_mal))
print("\n", classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

print("Threshold | Precision | Recall")
for t in [0.3, 0.5, 0.7, 0.9]:
    pred = np.where(p_mal >= t, 0, 1)
    print("   %.1f    |   %.3f   | %.3f" % (t, precision_score(y_test, pred, pos_label=0), recall_score(y_test,
pred, pos_label=0)))

fig, ax = plt.subplots(2, 1, figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
ax[0].set_title("Confusion Matrix"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
fpr, tpr, _ = roc_curve(y_test == 0, p_mal)
ax[1].plot(fpr, tpr, label="AUC = %.3f" % roc_auc_score(y_test == 0, p_mal)); ax[1].plot([0, 1], [0, 1], "--",
color="gray")
ax[1].set_title("ROC Curve"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()
plt.tight_layout()

# EXPERIMENT NO. 07 – SVM with Linear and Nonlinear Kernels 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                              classification_report)

wine = load_wine()
X, y = wine.data, wine.target
print("Shape:", X.shape, "| Class counts:", pd.Series(y).value_counts().sort_index().to_dict())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

kernels = {"Linear": SVC(kernel="linear", C=1),
           "Polynomial": SVC(kernel="poly", degree=3, coef0=1, C=1),
           "RBF": SVC(kernel="rbf", C=1, gamma="scale")}
rows, preds = [], {}
for name, m in kernels.items():
    m.fit(X_train, y_train); p = m.predict(X_test); preds[name] = p
    P, R, F, _ = precision_recall_fscore_support(y_test, p, average="macro")
    rows.append([name, accuracy_score(y_test, p), P, R, F, m.n_support_.sum()])
print("\n", pd.DataFrame(rows, columns=["Kernel", "Accuracy", "Precision", "Recall", "F1", "Support Vectors"])
      .round(4).to_string(index=False))
print("\nClassification report (RBF):\n", classification_report(y_test, preds["RBF"]))

print("Effect of C and gamma (RBF, 5-fold CV accuracy):")
for C, g in [(0.1, 0.01), (1, 0.1), (10, 0.1), (10, 10)]:
    acc = cross_val_score(SVC(kernel="rbf", C=C, gamma=g), X_train, y_train, cv=5).mean()
    print(" C = %-4s gamma = %-5s -> %.4f" % (C, g, acc))

fig, ax = plt.subplots(3, 1, figsize=(8, 12))
for a, (name, p) in zip(ax, preds.items()):
    sns.heatmap(confusion_matrix(y_test, p), annot=True, cmap="Blues", cbar=False, ax=a)
    a.set_title("%s (acc %.3f)" % (name, accuracy_score(y_test, p))); a.set_xlabel("Predicted")
ax[0].set_ylabel("Actual"); plt.tight_layout()

# EXPERIMENT NO. 08 – K-Means Clustering with Optimal Number of Clusters 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)         
print("Shape:", X.shape)

inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(1, 11)]
print("\n k | Inertia (WCSS)")
for k, v in zip(range(1, 11), inertia): print("%2d | %.2f" % (k, v))

sil = {k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)) for k in range(2, 11)}
print("\n k | Silhouette")
for k, v in sil.items(): print("%2d | %.4f" % (k, v))

k_opt = 3                                             
km = KMeans(n_clusters=k_opt, n_init=10, random_state=42).fit(X)
print("\nOptimal k (Elbow) = 3 | Highest silhouette at k =", max(sil, key=sil.get))
print("Cluster sizes:", pd.Series(km.labels_).value_counts().sort_index().to_dict())
print("\nClusters vs true species (validation only):")
print(pd.crosstab(km.labels_, iris.target_names[iris.target], rownames=["cluster"], colnames=["species"]))

Z = PCA(2).fit(X); P = Z.transform(X); C = Z.transform(km.cluster_centers_)
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(range(1, 11), inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k")
ax[1].plot(list(sil), list(sil.values()), "o-", color="#1baf7a"); ax[1].set_title("Silhouette Score");
ax[1].set_xlabel("k")
ax[2].scatter(P[:, 0], P[:, 1], c=km.labels_, cmap="viridis", s=12)
ax[2].scatter(C[:, 0], C[:, 1], marker="X", s=150, color="red"); ax[2].set_title("Clusters (k = 3, PCA view)")
plt.tight_layout()

# EXPERIMENT NO. 09 – DBSCAN – Density-Based Spatial Clustering of Applications with Noise 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)
print("Shape:", X.shape)

db = DBSCAN(eps=2.3, min_samples=4).fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
mask = labels != -1

print("\neps = 2.3, min_samples = 4")
print("Number of clusters        :", n_clusters)
print("Number of noise points    :", n_noise)
print("Largest cluster size      :", pd.Series(labels[mask]).value_counts().max())
print("Silhouette (non-noise)    : %.4f" % silhouette_score(X[mask], labels[mask]))
print("Core points               :", len(db.core_sample_indices_))

print("\nEffect of eps and min_samples:")
print(" eps min_samples clusters noise")
for eps in [1.8, 2.0, 2.3, 2.6, 3.0]:
    for ms in [4, 8]:
        l = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        print(" %.1f      %2d         %d       %3d" % (eps, ms, len(set(l)) - (1 if -1 in l else 0), (l == -1).sum()))

P = PCA(2).fit_transform(X)
fig, ax = plt.subplots(figsize=(8, 4.5))
for c in range(n_clusters):
    ax.scatter(P[labels == c, 0], P[labels == c, 1], s=14, label="Cluster %d" % c)
ax.scatter(P[labels == -1, 0], P[labels == -1, 1], s=25, marker="x", color="black", label="Noise")
ax.set_title("DBSCAN clusters (PCA 2-D view)"); ax.legend(fontsize=7); plt.tight_layout()

# EXPERIMENT NO. 10 – Ensemble Learning: Random Forest and Boosting 
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)          # malignant (label 0) = positive class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", X_train.shape, "| Test:", X_test.shape)

models = {"Logistic Regression (Baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
          "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
          "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)}
rows = []
for name, m in models.items():
    t0 = time.perf_counter(); m.fit(X_train, y_train); t = time.perf_counter() - t0
    p = m.predict(X_test)
    rows.append([name, accuracy_score(y_test, p), precision_score(y_test, p, pos_label=0),
                 recall_score(y_test, p, pos_label=0), f1_score(y_test, p, pos_label=0), t])
res = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision(M)", "Recall(M)", "F1(M)", "Train time (s)"])
print("\n", res.round(4).to_string(index=False))
print("\nBest model (by malignant recall / F1):", res.loc[res["F1(M)"].idxmax(), "Model"])

print("\nEffect of number of trees in Random Forest:")
print(" Trees Accuracy Time (s)")
for n in [1, 10, 50, 100, 200, 500]:
    t0 = time.perf_counter(); rf = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
    print(" %4d    %.4f    %.3f" % (n, accuracy_score(y_test, rf.predict(X_test)), time.perf_counter() - t0))

ax = res.set_index("Model")[["Accuracy", "Precision(M)", "Recall(M)", "F1(M)"]].plot(kind="bar", figsize=(8, 4.5),
rot=0, width=0.75)
ax.set_ylim(0.85, 1.0); ax.set_title("Model comparison"); ax.legend(ncol=4, fontsize=7, loc="lower center");
plt.tight_layout()

# EXPERIMENT NO. 11 – Reinforcement Learning: Q-Learning in a Grid World 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N, START, GOAL, PITS = 4, (0, 0), (3, 3), {(1, 1), (2, 3)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          
ARROWS = ["↑", "↓", "←", "→"]

def step(pos, a):
    r, c = pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]
    if not (0 <= r < N and 0 <= c < N): r, c = pos    
    if (r, c) == GOAL: return (r, c), 10, True
    if (r, c) in PITS: return (r, c), -10, True
    return (r, c), -1, False

def q_learning(episodes=500, alpha=0.1, gamma=0.9, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N * N, 4)); rewards, success = [], []
    for ep in range(episodes):
        pos, total, done = START, 0, False
        for t in range(100):
            s = pos[0] * N + pos[1]
            a = rng.integers(4) if rng.random() < eps else rng.choice(np.flatnonzero(Q[s] == Q[s].max()))
            pos, r, done = step(pos, a); s2 = pos[0] * N + pos[1]
            Q[s, a] += alpha * (r + (0 if done else gamma * Q[s2].max()) - Q[s, a])  
            total += r
            if done: break
        rewards.append(total); success.append(pos == GOAL)
    return Q, np.array(rewards), np.array(success)

Q, rewards, success = q_learning(eps=0.1)
print("Q-table (rows = states, cols = Up, Down, Left, Right):")
print(pd.DataFrame(Q, columns=["Up", "Down", "Left", "Right"]).round(2).to_string())

policy = Q.argmax(axis=1)
print("\nOptimal (greedy) policy:")
for r in range(N):
    print(" " + " ".join("G" if (r, c) == GOAL else "P" if (r, c) in PITS else ARROWS[policy[r * N + c]] for c
in range(N)))

print("\nStrategy           Avg Reward Success % Avg reward (last 100)")
curves = {}
for name, e in [("High (eps=0.5)", 0.5), ("Balanced (eps=0.1)", 0.1), ("Low (eps=0.01)", 0.01)]:
    runs = [q_learning(eps=e, seed=s) for s in range(20)]
    R = np.array([r for _, r, _ in runs]); S = np.array([s for _, _, s in runs])
    curves[name] = R.mean(axis=0)
    print("%-18s %8.2f   %7.2f   %8.2f" % (name, R.mean(), S.mean() * 100, R[:, -100:].mean()))

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, c in curves.items():
    ax.plot(np.convolve(c, np.ones(15) / 15, mode="valid"), label=name)
ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Learning curve (avg of 20 runs)"); ax.legend();
plt.tight_layout()
