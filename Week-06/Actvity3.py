import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
# 1. Load Dataset fom the Kaggle HousPrice
house_data= pd.read_csv("train.csv")

print("Dataset shape:", house_data.shape)
print("Printing First 10 rows:\n", house_data.head(10))
print("Missing Values:",house_data.isnull().sum())
print("House Price Data Summary Statistics of Slaes:\n", house_data['SalePrice'].describe())


#Separate X and Y 
y_target = house_data['SalePrice'].values
X_feature = house_data.drop(columns=['SalePrice','Id'])

#Handle if any missing values and convert categorical features into numerical
x_f=pd.get_dummies(X_feature)
X_f=x_f.fillna(x_f.mean())

#Train test split
x_train,x_test,y_train,y_test = train_test_split(X_f,y_target,test_size=0.2,random_state=42)

#Training   rnadom classifier using default parameters
rf_model1 =RandomForestRegressor(random_state=42,n_jobs=-1)
rf_model1.fit(x_train,y_train)
model_pred = rf_model1.predict(x_test)
rmse_error = np.sqrt(mean_squared_error(y_test,model_pred))
r2_error = r2_score(y_test,model_pred)
print(f"Basic Model Test RMSE: {rmse_error:.4f}, R2 Error: {r2_error:.4f}")
#Definig Parameter gird for girdSearchCV
param_grid = {"n_estimators":[100,200],"max_depth":[None,10,20],"min_samples_split":[2,5],"min_samples_leaf":[1,2],"max_features":[0.7,1.0],}

#Grid Search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42,n_jobs=-1),param_grid=param_grid,cv=5,scoring="neg_mean_squared_error",n_jobs=-1)
grid_search.fit(x_train,y_train)

best_cv_mse_error = -grid_search.best_score_
best_cv_rmse_error = np.sqrt(best_cv_mse_error)
print("\nBest Hyperparameters:\n", grid_search.best_params_)
print(f"Best CV RMSE: {best_cv_rmse_error:.4f}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)

mae_error = mean_absolute_error(y_test,y_pred)
mse_error = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse_error)
r2_s = r2_score(y_test,y_pred)

print(f"Test MAE: {mae_error:.4f}")
print(f"Test MSE: {mse_error:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R² Score: {r2_s:.4f}")

comparison_table = pd.DataFrame({
        "Model": ["Default Random Forest", "Optimized Random Forest"],
        "Test RMSE": [rmse_error, rmse],
        "Test R² Score": [r2_error, r2_s],
    }
)
print("\n", comparison_table)


plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.7,color='darkorange')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"k--",lw=2,)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(alpha=0.3)
plt.show()

#Important Fetures three most 
important_feat = pd.Series(best_rf.feature_importances_,index=X_feature.columns)
plt.figure(figsize=(8, 5))
important_feat.nlargest(10).plot(kind="barh", color="teal")
plt.title("Top 5 influential features in House price")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.show()
print(
    "Top 5 Influential Features:\n",
    important_feat.nlargest(5),
)