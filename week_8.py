import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset from seaborn library
df = sns.load_dataset('tips')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Selecting features (Income-like and Spending-like)
X = df[['total_bill', 'tip']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# -------------------------------
# Apply K-Means (choose k = 3)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# -------------------------------
# Plot Clusters
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1],
            s=250, c='red', marker='X', label='Centroids')

plt.xlabel("Total Bill (Scaled)")
plt.ylabel("Tip (Scaled)")
plt.title("Customer Clusters using K-Means")
plt.legend()
plt.show()

# -------------------------------
# Cluster Analysis
# -------------------------------
print("\nCluster-wise Average Values:")
print(df.groupby('Cluster')[['total_bill', 'tip']].mean())

print("\nNumber of Customers in Each Cluster:")
print(df['Cluster'].value_counts())
