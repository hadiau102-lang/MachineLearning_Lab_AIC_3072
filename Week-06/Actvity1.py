import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset fom the Kaggle fashoion_mnist
fashion_data= pd.read_csv("fashion-mnist_train.csv")

print("Dataset shape:", fashion_data.shape)
print("Printing First 10 rows:\n", fashion_data.head(10))
print("Fashion Data Class distribution:\n", fashion_data['label'].value_counts())

#Feature and target values
y_target = fashion_data['label'].values
X_feature = fashion_data.drop(columns=['label']).values


#Applying Standard Scaling for normalization
std_scaler = StandardScaler()
x_scaled = std_scaler.fit_transform(X_feature)
#Applying PCA using three different approaches
# Approach 1: Fixed number of components telling him to divide into how many pca componenets like 2 or 3 or so on
pca_50_comp = PCA(n_components=50,random_state=42)
x_pca1_approach = pca_50_comp.fit_transform(x_scaled)

# Approach 2: Telling by percentage of varinace like to that percnetgae apply pca
pca_95_percent = PCA(n_components=0.95,random_state=42)
x_pca2_approach = pca_95_percent.fit_transform(x_scaled)

# Method 3: Now applied pca on all total no of features availble like if 13 features so on  13 all
pca_all_comp = PCA(n_components=None,random_state=42)
x_pca3_approach = pca_all_comp.fit_transform(x_scaled)

# Now detailed summary of this all
print("\nPCA Summary ")
print(f"Appraoch 1 (Fixed componenets 50): Original features: {X_feature.shape[1]}, Retained ones: 50, Cumulative Variance is: {np.sum(pca_50_comp.explained_variance_ratio_):.4f}")
print(f"Appraoch 2 (Fixed variance 95%): Original features: {X_feature.shape[1]}, Retained ones: 95%, Cumulative Variance is: {np.sum(pca_95_percent.explained_variance_ratio_):.4f}")
print(f"Appraoch 3 (No fixed Components applied on all): Original features: {X_feature.shape[1]}, Retained ones: All, Cumulative Variance is: {np.sum(pca_all_comp.explained_variance_ratio_):.4f}")

# Ploting the  Cumulative Explained Variance now
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_all_comp.explained_variance_ratio_), marker="o",linestyle="--", color="b",)
plt.xlabel("Total Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance VS Total Number of Components")
plt.grid(True)
plt.show()

#  Visualize dataset into  2D using first two principal components
pca_2d_visulize = PCA(n_components=2, random_state=42)
x_2d = pca_2d_visulize.fit_transform(x_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y_target, cmap="tab10", alpha=0.6, edgecolor="none")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("2D PCA Visulaization for  Fashion MNIST Data")
plt.colorbar(scatter, label="Clothing Class")
plt.grid(alpha=0.3)
plt.show()

#  Comparison Table
comparison_table= pd.DataFrame(
    {"Approach": ["1 (Fixed Components (50))","2(95% Variance)","3(All Components)",],
     "Original Features of Fasion Dataset": [X_feature.shape[1]] * 3,
    "Retained Components": [50,pca_95_percent.n_components_,pca_all_comp.n_components_,],
    "Cumulative Expalined Variance": [np.sum(pca_50_comp.explained_variance_ratio_),np.sum(pca_95_percent.explained_variance_ratio_),np.sum(pca_all_comp.explained_variance_ratio_),]})
print("\n", comparison_table)