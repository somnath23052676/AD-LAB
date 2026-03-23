import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

wine = load_wine()
X = wine.data
y = wine.target

X = StandardScaler().fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X)

print("Explained Variance Ratio:")
for i,v in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {v:.4f}")

pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(X)

plt.figure()
for i in np.unique(y):
    plt.scatter(X2[y==i,0], X2[y==i,1], label=f"Class {i}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection")
plt.legend()
plt.show()

pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in np.unique(y):
    ax.scatter(X3[y==i,0], X3[y==i,1], X3[y==i,2], label=f"Class {i}")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D Projection")
plt.legend()
plt.show()
