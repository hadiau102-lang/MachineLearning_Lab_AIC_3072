#Hadia Usman
#23-NTU-CS-1029
#BSCS 7th A


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Download the Dataset from Kaggle Online
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Dataset downloaded to path:", path)

# Locate the directory containing the 'yes' and 'no' subfolders
data_dir = path
for root, dirs, files in os.walk(path):
    if 'yes' in dirs and 'no' in dirs:
        data_dir = root
        break

print("Image data directory located at:", data_dir)

#  Load and Preprocess MRI Images
IMG_SIZE = 64
images = []
labels = []

categories = ['no', 'yes']  
# 'no' = 0 (No Tumor), 'yes' = 1 (Tumor)
for category in categories:
    folder_path = os.path.join(data_dir, category)
    if not os.path.exists(folder_path):
        continue
    class_idx = categories.index(category)
    
    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            # Convert to grayscale
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is not None:
                # Resize to 64x64
                img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                # Flatten each image into a 1D feature vector
                img_flattened = img_resized.flatten()
                images.append(img_flattened)
                labels.append(class_idx)
        except Exception as e:
            pass

X_feature = np.array(images)
y_target = np.array(labels)

print(f"Total images loaded: {X_feature.shape[0]}")
print(f"Feature vector size per image: {X_feature.shape[1]}")

#  Split data into training and testing sets (80:20 ratio)
x_train, x_test, y_train, y_test = train_test_split(
    X_feature, y_target, test_size=0.2, random_state=42, stratify=y_target
)

#  StandardScaler()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#  Training
print("\nTraining SVM (RBF Kernel) model...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(x_train_scaled, y_train)

# Evaluate Performance
y_pred = svm_model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# Plot Confusion Matrix using pure Matplotlib
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - SVM Brain Tumor Detection')
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, ['No Tumor', 'Tumor'])
plt.yticks(tick_marks, ['No Tumor', 'Tumor'])

# Add text annotations inside the confusion matrix blocks
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()  
#  Display a few test images with predicted and actual labels
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.ravel()

for i in range(8):
    if i < len(x_test):
        # Reshape back to 64x64 for image plotting
        img_vis = x_test[i].reshape(IMG_SIZE, IMG_SIZE)
        pred_label = categories[y_pred[i]]
        true_label = categories[y_test[i]]
        
        axes[i].imshow(img_vis, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        axes[i].axis('off')

plt.tight_layout()
plt.show()