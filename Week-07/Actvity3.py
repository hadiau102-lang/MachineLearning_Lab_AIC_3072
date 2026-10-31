#Hadia Usman 
#23-NTU-CS-1029
#BSCS 7th A 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import Ridge , LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,StackingRegressor

import warnings 
warnings.filterwarnings('ignore')

insurance_data = pd.read_csv('Medical Cost Personal Datasets.csv')

print("Data Shape:",insurance_data.shape)
print("\nFirst 10 rows:",insurance_data.head(10))
print("\nMissing Values:",insurance_data.isnull().sum())
print("\nStatistics of Charges  :",insurance_data['charges'].describe())

insurance_data['sex']=insurance_data['sex'].map({'female':0 , 'male':1})
insurance_data['smoker']=insurance_data['smoker'].map({'no':0 , 'yes':1})


insurance_data_enc=pd.get_dummies(insurance_data,columns=['region'],drop_first=True)
X_feature=insurance_data_enc.drop('charges',axis=1)
Y_target=insurance_data_enc['charges']
x_train ,x_test, y_train,y_test = train_test_split(X_feature,Y_target,test_size=0.25,random_state=20)


scaler=StandardScaler()

X_trained_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
X_train_scaled_df = pd.DataFrame(X_trained_scaled, columns=X_feature.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_feature.columns)

ridge_model = Ridge(alpha=0.1,random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
grd_class = GradientBoostingRegressor(random_state=42)


ridge_model.fit(X_train_scaled_df,y_train)
rf_reg.fit(x_train,y_train)
grd_class.fit(x_train,y_train)

estimatros = [('ridge', Ridge(alpha=0.1,random_state=42)),
          ('rf',RandomForestRegressor( random_state=42)),
          ('gb', GradientBoostingRegressor(random_state=42))]

stack_reg = StackingRegressor(estimators=estimatros,final_estimator=LinearRegression(),cv=5)
stack_reg.fit(x_train,y_train)

models = {
    "Ridge": (ridge_model, X_test_scaled_df),
    "Random Forest": (rf_reg, x_test),
    "Gradient Boosting": (grd_class, x_test),
    "Stacking regressor": (stack_reg, x_test),
}

results = []
for name, (model, data) in models.items():
    preds = model.predict(data)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2 Score": r2_score(y_test, preds)
    })


results_df = pd.DataFrame(results)
print("\n Model Comparison Table ")
print(results_df)

print("\nFinal Estimator (Linear Regression) Coefficients ")
for name, coef in zip([est[0] for est in estimatros], stack_reg.final_estimator_.coef_):
    print(f"{name}: {coef:.4f}")

stack_prediction = stack_reg.predict(x_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, stack_prediction, alpha=0.5, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Charges are")
plt.ylabel("Predicted Charges are")
plt.title("Stacking Regressor presenting  Actual vs Predicted Charges")
plt.show()