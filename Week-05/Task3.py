#k-meansS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load your existing CSV file
# (Adjust file path to "/content/mushrooms.csv" if your file has a single 's')
file_path = "mushroomss.csv"
mushroom_data = pd.read_csv(file_path)

# Clean column names to remove any accidental whitespace
mushroom_data.columns = mushroom_data.columns.str.strip()

print("Dataset Preview:")
print(mushroom_data.head())
print(f"\nDataset shape: {mushroom_data.shape}")

# Step 2: Convert Categorical Strings to Numeric Features
# This loops through every column in your file and encodes 'x', 's', 'n' into numbers
X = mushroom_data.copy()
le = LabelEncoder()
for column in X.columns:
    X[column] = le.fit_transform(X[column])

# Step 3: Run Unsupervised K-Means Clustering
# Grouping into 2 clusters (commonly edible vs poisonous in mushroom datasets)
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Step 4: Reduce Dimensions using PCA for Plotting
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Step 5: Visualize the Clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('Mushroom Data Clusters (K-Means + PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Assignment')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()