import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
# 1. Load Dataset fom the Kaggle Heart ine
heart_data= pd.read_csv("heart.csv")

print("Dataset shape:", heart_data.shape)
print("Printing First 10 rows:\n", heart_data.head(10))
print("Missing Values:",heart_data.isnull().sum())
print("Heart Data Target distribution:\n", heart_data['target'].value_counts())

#Cleaning dataset and encoding if needed
heart_data=heart_data.dropna()
#Separate X and Y 
y_target = heart_data['target'].values
X_feature = heart_data.drop(columns=['target'])

#Train test split
x_train,x_test,y_train,y_test = train_test_split(X_feature,y_target,test_size=0.2,random_state=42,stratify=y_target)

#Training   rnadom classifier using default parameters
rf_model1 =RandomForestClassifier(random_state=42)
rf_model1.fit(x_train,y_train)
model_pred = rf_model1.predict(x_test)
print(f"basic Random Forest Test Accuracy: {accuracy_score(y_test, model_pred):.4f}")

#Definig Parameter gird for girdSearchCV
param_grid = {"n_estimators":[100,200],"max_depth":[None,10,20],"min_samples_split":[2,5],"min_samples_leaf":[1,2],"max_features":['sqrt','log2']}

#Grid Search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid_search.fit(x_train,y_train)

print("\nBest Hyperparameters:\n", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)


print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

#Confusion Matrix
cm_matrix = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Heart Disease Dataset")
plt.xlabel("Predicted Label is ")
plt.ylabel("Actual Label is")
plt.show()


#Important Fetures three most 
important_feat = pd.Series(best_rf.feature_importances_,index=X_feature.columns)
plt.figure(figsize=(8, 5))
important_feat.nlargest(10).plot(kind="barh", color="teal")
plt.title("Random Forest Feature Importances")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.show()
print("\nTop 3 Most Influential Features:\n", important_feat.nlargest(3))