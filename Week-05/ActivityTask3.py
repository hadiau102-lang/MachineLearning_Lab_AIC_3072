# Hadia Usman
# 23-NTU-CS-1029
# BS CS 7th A

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

csv_file = "Mall_Customers.csv"
df = pd.read_csv(csv_file)

print("\n First 5 rows of the Dataset")
print(df.head())

# Extract 'Annual Income (k$)' and 'Spending Score (1-100)' for clustering
X_feature= df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Apply K-Means & DBSCAN (Unscaled) and plot clusters
print("\nRunning K-Means and DBSCAN (Unscaled) First")

# K-Means (arbitrary k=5 as standard for this dataset)
kmeans_unscaled = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans_unscaled = kmeans_unscaled.fit_predict(X_feature)

# DBSCAN
dbscan = DBSCAN(eps=5.0, min_samples=5)
y_dbscan = dbscan.fit_predict(X_feature)

# Plotting Unscaled K-Means vs DBSCAN
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_feature[:, 0], X_feature[:, 1], c=y_kmeans_unscaled, cmap='viridis', s=50, edgecolors='k')
axes[0].set_title('K-Means (Unscaled) - Annual Income vs Spending Score')
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score (1-100)')

axes[1].scatter(X_feature[:, 0], X_feature[:, 1], c=y_dbscan, cmap='plasma', s=50, edgecolors='k')
axes[1].set_title('DBSCAN (Unscaled) - Annual Income vs Spending Score')
axes[1].set_xlabel('Annual Income (k$)')
axes[1].set_ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

# Scale features using StandardScaler and re-run K-Means
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_feature)

kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans_scaled = kmeans_scaled.fit_predict(x_scaled)

# Compare both results visually & conceptually
plt.figure(figsize=(7, 6))
plt.scatter(X_feature[:, 0], X_feature[:, 1], c=y_kmeans_scaled, cmap='viridis', s=50, edgecolors='k')
plt.title('K-Means (Scaled) - Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

print("\n---  Comparison ---")
print("Did the scaling change clusters? On this specific 2D dataset where both features share similar numerical ranges (First one is Income: 15-140, and the other is  Spending: 1-100),It may be that by doing scaling it might not dramatically shift clusters visually. However, scaling ensures that features with larger numerical ranges don't overpower others when calculating Euclidean distances.")

# Using the Elbow Method to find the best k
inertia = []
K_ranges = range(1, 11)
for k in K_ranges:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(x_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K_ranges, inertia, marker='o', linestyle='-', color='b')
plt.title('The Elbow Method to Find Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

print("\n--- Task 5 ---")
print("The 'elbow' point on the graph (where the curve bends and flattening begins) indicates the optimal k, which is typically k = 5 for this dataset.")

#   interpreting each cluster (using k=5 scaled model centroids)
print("\n--- Task 6: Cluster Interpretations (k=5) ---")
centers_original = scaler.inverse_transform(kmeans_scaled.cluster_centers_)
for i, center in enumerate(centers_original):
    print(f"Cluster {i}: Average Annual Income = ${center[0]:.2f}k | Average Spending Score = {center[1]:.2f}")

print("\nCluster Interpretation Profiles:")
print("- Low Income, High Spending (Careless spenders)")
print("- Low Income, Low Spending (Cautious/Budget savers)")
print("- Moderate Income, Moderate Spending (Average mainstream customers)")
print("- High Income, High Spending (Target/Premium high-spenders)")
print("- High Income, Low Spending (Prudent wealthy savers)")

#  Adding a fake feature "LuckyNumber" and explain its effect
np.random.seed(42)
df['LuckyNumber'] = np.random.randint(0, 101, size=len(df))
X_with_lucky = np.column_stack((x_scaled, df['LuckyNumber'].values))

kmeans_lucky = KMeans(n_clusters=5, random_state=42, n_init=10)
y_lucky = kmeans_lucky.fit_predict(X_with_lucky)

print("\n Effect of adding 'LuckyNumber' with (Random Feature)")
print("Since 'LuckyNumber' is purely random uniform noise (0-100), adding it introduces irrelevant variance into distance calculations. This distorts the true natural groupings, causes cluster boundaries to blur, and reduces cluster quality. This demonstrates why feature selection is vital before clustering!")