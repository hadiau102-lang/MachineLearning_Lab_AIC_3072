#23-NTU-CS-1029 Hadia Usman  BSCS 7th A
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score,f1_score , mean_squared_error


#Load wind dataset 
credit_card_data = pd.read_csv("creditcard.csv",encoding='latin-1')
credit_card_data = credit_card_data.sample(n=10000, random_state=42)
X_credit_Card_features = credit_card_data.drop(columns=['Class'])
y_target = credit_card_data['Class']

#Train Test Split
x_train , x_test , y_train, y_test = train_test_split(X_credit_Card_features,y_target,test_size=0.2,random_state=42,stratify=y_target)


#Imabalnced dataset 
#Smote applied

smote_data =SMOTE(random_state=42)
x_train_resample,y_train_resampled = smote_data.fit_resample(x_train,y_train)

#Standard scale so all features have equal priorty
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(x_train_resample)
X_test_scaled = scalar.transform(x_test)

#Train and test a Knn Classifier for each value of k and tune k using GirdSerach CV

gird_parameters = {'n_neighbors':[3,5,7]}
gird_search = GridSearchCV(estimator=  KNeighborsRegressor(),param_grid=gird_parameters,scoring='neg_mean_squared_error',cv=3)
gird_search.fit(X_train_scaled,y_train_resampled)

#Best model
best_model = gird_search.best_estimator_
print(f"Best k: {gird_search.best_params_['n_neighbors']}")

#Evaluate perfomance using accuracy , F1 score
y_pred = best_model.predict(X_test_scaled)
y_pred_bin = (y_pred >= 0.5 ).astype(int)

print(f"Accuracy : {accuracy_score(y_test,y_pred_bin):.4f}")
print(f"F1 Score : {f1_score(y_test,y_pred_bin):.4f}")
