#Hadia Usman 
#23-NTU-CS-1029
#BSCS 7th A
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
import warnings 
warnings.filterwarnings('ignore')


telco_data = pd.read_csv('Telco Customer Churn.csv')

print("Data Shape:",telco_data.shape)
print("\nFirst 10 rows:",telco_data.head(10))
print("\nMissing Values:",telco_data.isnull().sum())
print("\nClass Distribution :",telco_data['Churn'].value_counts())

telco_data['TotalCharges']=pd.to_numeric(telco_data['TotalCharges'],errors='coerce')

telco_data['TotalCharges'].fillna(telco_data['TotalCharges'].median(),inplace=True)

telco_data=telco_data.drop('customerID',axis=1)
telco_data['Churn']=telco_data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

telco_data_enc=pd.get_dummies(telco_data,drop_first=True)
telco_data_enc = telco_data_enc.fillna(telco_data_enc.median())
X_feature=telco_data_enc.drop('Churn',axis=1)
Y_target=telco_data_enc['Churn']
x_train ,x_test, y_train,y_test = train_test_split(X_feature,Y_target,test_size=0.25,random_state=20)



grd_class = GradientBoostingClassifier(random_state=42)
ada_class = AdaBoostClassifier(random_state=42)
xgb_class = XGBClassifier(random_state=42)


grd_class.fit(x_train,y_train)
ada_class.fit(x_train,y_train)
xgb_class.fit(x_train,y_train)


param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search.fit(x_train,y_train)
print("\n Best parameters for xgbbost")
print(grid_search.best_params_)
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_

def evalute_models(model,X_te,y_te):
    preds = model.predict(X_te)
    probs= model.predict_proba(X_te)[:,1]
    return({
    
        "Accuracy": accuracy_score(y_te, preds),
        "Precision": precision_score(y_te, preds),
        "Recall": recall_score(y_te, preds),
        "F1-Score": f1_score(y_te, preds),
        "ROC_AUC":roc_auc_score(y_te,probs)
    })

eval= evalute_models(xgb_class,x_test,y_test)

eval_tuned= evalute_models(best_xgb,x_test,y_test)

comparison_table = pd.DataFrame([eval, eval_tuned], 
                             index=["Default XGBoost", "Tuned XGBoost"])
print("\n--- Default vs Tuned XGBoost ---")
print(comparison_table)




feat_importances=pd.Series(best_xgb.feature_importances_, index=X_feature.columns)
top_5_features = feat_importances.nlargest(5)
print("\n 5 Top Most Important Features ")
print(top_5_features)

plt.figure(figsize=(10, 6))
top_5_features.plot(kind='barh',color='teal')
plt.title("Top 5 important Features")
plt.xlabel("Realtive Importance")
plt.ylabel("Feature")
plt.show()