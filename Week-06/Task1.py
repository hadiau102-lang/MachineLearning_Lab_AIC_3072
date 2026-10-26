#PCA 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine_data = load_wine()

X_feature = wine_data.data
Y_target = wine_data.target

feat_names = wine_data.feature_names
target_names = wine_data.target_names

print("Original dataset shape:",X_feature.shape)
print("Target shape:",Y_target.shape)
print("Feature names:",feat_names)
print("Target names:",target_names)

scaler= StandardScaler()
x_scaled = scaler.fit_transform(X_feature)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(x_scaled)

print("Original Dataset Shape:",X_feature.shape)
print("Reduced Dataset Shape:",X_reduced.shape)

wine_colors =['orangered','darkred','crimson']

plt.figure(figsize=(10,6))

for class_num , color, target_name in zip(
  [0,1,2], wine_colors ,target_names
  ):

  plt.scatter( X_reduced[Y_target == class_num,0],
               X_reduced[Y_target == class_num,1],
               color=color,
               label=target_name,
               alpha=0.7)

plt.title("PCA of Wine Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Componenet")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("Principal component loadings:")
print(pca.components_)
print("Shape of principal components:",pca.components_.shape)

plt.figure(figsize=(14,5))
sns.heatmap(
  pca.components_,
  cmap='viridis',
  xticklabels=feat_names,
  yticklabels=['First Componenet','Second Component']
)

plt.title('PCA Componenets Loadings')
plt.xlabel("Orginal features")
plt.ylabel("Principal Components")
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print("\nExplained varaince ratio:")
print(pca.explained_variance_ratio_)


pca_3 = PCA(n_components=0.96)
X_reduced_96=pca_3.fit_transform(x_scaled)

print("Number of componenets selected:", pca_3.n_components_)

print("Total Explained Variance:",
      np.sum(pca_3.explained_variance_ratio_))
print("Reduced Dataset shape:",X_reduced_96.shape)