#Hadia Usman , 23-NTU-CS-1029 , BSCS 7th A
#-------------------Class Task From A to D-------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, TimeSeriesSplit, GroupKFold, cross_val_predict, cross_val_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from imblearn.over_sampling import SMOTE
#Import libraraies and load data A part 1

raw_data=pd.read_csv('titanic.csv')
feature_columns = ['Pclass','Gender','Age','SibSp','Parch','Fare']
data_titanic= raw_data[['Survived']+feature_columns].copy()

group_labels = raw_data.loc[data_titanic.index, "Ticket"].fillna(
    pd.Series(data_titanic.index.astype(str), index=data_titanic.index)).astype(str)


encoder = LabelEncoder()
data_titanic["Gender"] = encoder.fit_transform(data_titanic["Gender"].astype(str))

age_imputer = SimpleImputer(strategy="median")
data_titanic[["Age"]] = age_imputer.fit_transform(data_titanic[["Age"]])

X_f = data_titanic.drop("Survived", axis=1)
y_t = data_titanic["Survived"]

print("Dataset shape:", data_titanic.shape)
print("Original CSV shape:", raw_data.shape)
print("Feature columns:", list(X_f.columns))
print(data_titanic.head())

print("\nSurvival counts:")
print(y_t.value_counts())

print("\nMissing values after data preprocessing:")
print(data_titanic.isna().sum())

print("\nNumber of ticket groups are:", group_labels.nunique())

#Tarin , Validation and Test Split A part 2
X_temp, X_test, y_temp, y_test = train_test_split(
    X_f, y_t, test_size=0.20, random_state=42, stratify=y_t
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

depth_values = [2, 4, 6, 8, None]
results = []

for depth in depth_values:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    results.append([depth, train_accuracy, val_accuracy])

    split_results = pd.DataFrame(results, columns=["Max Depth", "Training Accuracy", "Validation Accuracy"])
print(split_results)

# Select the depth with the highest validation accuracy
best_depth = split_results.loc[
    split_results["Validation Accuracy"].idxmax(), "Max Depth"
]
if pd.isna(best_depth):
    best_depth = None
print("Best depth:", best_depth)

# Train the selected model and evaluate it on the test data
best_model = RandomForestClassifier(max_depth=best_depth, random_state=42)
best_model.fit(X_train, y_train)

print("Final training accuracy:", accuracy_score(y_train, best_model.predict(X_train)))
print("Final validation accuracy:", accuracy_score(y_val, best_model.predict(X_val)))
print("Final test accuracy:", accuracy_score(y_test, best_model.predict(X_test)))

# B Cross Validation and metrics 
folds_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

predictions_model = cross_val_predict(model, X_f, y_t, cv=folds_model)


probabilities = cross_val_predict(model, X_f, y_t, cv=folds_model, method="predict_proba")[:, 1]

# Confusion matrix 
tn, fp, fn, tp = confusion_matrix(y_t, predictions_model).ravel()

accuracy = accuracy_score(y_t, predictions_model)
precision = precision_score(y_t, predictions_model, zero_division=0)
recall = recall_score(y_t, predictions_model, zero_division=0)
f1 = f1_score(y_t, predictions_model, zero_division=0)
specificity = tn / (tn + fp)
auc = roc_auc_score(y_t, probabilities)

print("Confusion Matrix:")
print(confusion_matrix(y_t, predictions_model))
print("\nEvaluation Results")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))
print("Specificity:", round(specificity, 3))
print("ROC-AUC:", round(auc, 3))

#C Comparing Six splitting Methods
comparison_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

comparison_results = []

# 1. Hold-Out:  80 training and 20 testing split
X_train_ho, X_test_ho, y_train_ho, y_test_ho = train_test_split( X_f, y_t, test_size=0.20, random_state=42, stratify=y_t)
holdout_model = RandomForestClassifier(n_estimators=100, random_state=42)
holdout_model.fit(X_train_ho, y_train_ho)
holdout_accuracy = accuracy_score(y_test_ho, holdout_model.predict(X_test_ho))
comparison_results.append({
    "Method": "Hold-Out",
    "Mean Accuracy": holdout_accuracy,
    "Std Accuracy": 0.0,
    "Number of Test Splits": 1
})

# 2. K-Fold: five folds train and test on all data 
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(comparison_model, X_f, y_t, cv=kfold, scoring="accuracy")
comparison_results.append({
    "Method": "K-Fold",
    "Mean Accuracy": kfold_scores.mean(),
    "Std Accuracy": kfold_scores.std(),
    "Number of Test Splits": len(kfold_scores)
})
# 3. Stratified K-Fold: five folds with equal class ratios
stratified_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(comparison_model, X_f, y_t, cv=stratified_folds, scoring="accuracy")
comparison_results.append({
    "Method": "Stratified K-Fold",
    "Mean Accuracy": stratified_scores.mean(),
    "Std Accuracy": stratified_scores.std(),
    "Number of Test Splits": len(stratified_scores)
})
# 4. Repeated K-Fold for classification: repeat the stratified k fold 3 times or more 
repeated_folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
repeated_scores = cross_val_score(comparison_model, X_f, y_t, cv=repeated_folds, scoring="accuracy")
comparison_results.append({
    "Method": "Repeated K-Fold",
    "Mean Accuracy": repeated_scores.mean(),
    "Std Accuracy": repeated_scores.std(),
    "Number of Test Splits": len(repeated_scores)
})
# 5. Time Series Split: earlier rows train, later rows test.

time_folds = TimeSeriesSplit(n_splits=5)
time_scores = cross_val_score(comparison_model, X_f.reset_index(drop=True), y_t.reset_index(drop=True),cv=time_folds, scoring="accuracy")
comparison_results.append({
    "Method": "Time Series Split (row order demo)",
    "Mean Accuracy": time_scores.mean(),
    "Std Accuracy": time_scores.std(),
    "Number of Test Splits": len(time_scores)
})
# 6. Group K-Fold: all rows with the same ticket stay in one group
if group_labels.nunique() >= 5:
    group_folds = GroupKFold(n_splits=5)
    group_scores = cross_val_score(comparison_model, X_f, y_t, cv=group_folds,groups=group_labels, scoring="accuracy")
    comparison_results.append({
        "Method": "Group K-Fold (Ticket groups)",
        "Mean Accuracy": group_scores.mean(),
        "Std Accuracy": group_scores.std(),
        "Number of Test Splits": len(group_scores)
    })
else:
    print("Group K-Fold skipped: fewer than 5 unique groups are available.")

cv_comparison_table = pd.DataFrame(comparison_results)
print(cv_comparison_table.round(3))
# Visualizing comparison of each corss validation techniques accuracy 
plt.figure(figsize=(10, 5))
plt.bar(cv_comparison_table["Method"],cv_comparison_table["Mean Accuracy"])
plt.ylabel("Mean Accuracy")
plt.title("Comparison of Data-Splitting and Cross-Validation Methods")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
fpr, tpr, thresholds = roc_curve(y_t, probabilities)

#D Roc Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

print("ROC-AUC:", round(roc_auc_score(y_t, probabilities), 3))


#------------------Home Task From E to H ----------------------------------
#E classfication threshold comparison
threshold_results=[]
for threshold in np.arange(0.1,1.0,0.1):
    threshold_predictions = (probabilities >= threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_t,threshold_predictions).ravel()
    threshold_results.append(
        {
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_t, threshold_predictions),
        "Precision": precision_score(y_t, threshold_predictions, zero_division=0),
        "Recall": recall_score(y_t, threshold_predictions, zero_division=0),
        "F1-score": f1_score(y_t, threshold_predictions, zero_division=0),
        "Specificity": tn / (tn + fp)

        }
    )

threshold_table = pd.DataFrame(threshold_results)
print(threshold_table)
best_row=threshold_table.loc[threshold_table['F1-score'].idxmax()]
print("Best Threshold:",best_row['Threshold'])
print("Best F1-score:",round(best_row['F1-score'],3))

#F Overfitiing and UnderFitting
depths=[1,2,3,4,5,6,7,8,9,10,11,12,None]
fit_results=[]
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for depth in depths:
    tree=DecisionTreeClassifier(max_depth=depth,random_state=42)

    tree.fit(X_f,y_t)
    train_accuracy = accuracy_score(y_t,tree.predict(X_f))
    cv_accuracy=cross_val_score(tree,X_f,y_t,cv=folds).mean()

    fit_results.append({
      "Max Depth": str(depth),
      "Training Accuracy": train_accuracy,
      "Cross-validation Accuracy": cv_accuracy
    })
fit_table = pd.DataFrame(fit_results)
print(fit_table)

plt.figure(figsize=(8, 5))
plt.plot(fit_table["Max Depth"], fit_table["Training Accuracy"], marker="o", label="Training")
plt.plot(fit_table["Max Depth"], fit_table["Cross-validation Accuracy"], marker="o", label="Cross-validation")
plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting and Underfitting on Given Dataset")
plt.legend()
plt.grid(True)
plt.show()

#G Class Imbalanced
died=data_titanic[data_titanic['Survived']==0]
survivors = data_titanic[data_titanic['Survived']==1].sample(47,random_state=42)
imbalnced_data = pd.concat([died,survivors]).sample(frac=1,random_state=42)

X_imb = imbalnced_data.drop('Survived',axis=1)
Y_imb = imbalnced_data['Survived']
print("Class counts:")
print(Y_imb.value_counts())
print("Survivore percentage:",round(Y_imb.mean()*100,2),"%")

#Smote Demonstration
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
train_index,test_index=next(folds.split(X_imb,Y_imb))

X_train = X_imb.iloc[train_index]
Y_train = Y_imb.iloc[train_index]

print("Before SMOTE:")
print(Y_train.value_counts())

smote=SMOTE(random_state=42)
X_train_balanced,y_train_balanced=smote.fit_resample(X_train,Y_train)
print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

#H train and save the model

final_model=RandomForestClassifier(max_depth=best_depth,n_estimators=100,random_state=42)
final_model.fit(X_f,y_t)

model_bundle = {
    "model":final_model,
    "gender_encoder":encoder,
    "age_imputer":age_imputer,
    "feature_columns":list(X_f.columns),
    "target_name":"Survived"

}
joblib.dump(model_bundle,"titanic_random_forest_model.joblib")
print("Final model saved as titanic_random_forest_model.joblib")

loaded_bundle=joblib.load("titanic_random_forest_model.joblib")
loaded_model = loaded_bundle["model"]
print("Loaded model prediction for the first passenger:",loaded_model.predict(X_f.iloc[[0]]))
