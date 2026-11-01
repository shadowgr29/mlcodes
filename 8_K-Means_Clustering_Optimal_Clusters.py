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