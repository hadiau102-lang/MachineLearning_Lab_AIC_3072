
#23-NTU-CS-1029
#Hadia Usman
#BSCS 7th A 

#Activity 2
#Logistic Regression using Admission_Predict
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
data_admission = pd.read_csv('Admission_Predict.csv')
print(data_admission.head())
print(data_admission.columns)

#Encoding Label and One He Encoding

data_admission['Admission']=data_admission['Admission'].apply(
    lambda x: 'Admitted' if x >= 0.75 else 'Rejected'
)
Labelencoder = LabelEncoder()
y_target = Labelencoder.fit_transform(data_admission['Admission'])

X_features = data_admission[['GRE Score','TOEFL Score','CGPA','Research']]

#Train test 
x_train , x_test , y_train, y_test = train_test_split(X_features,y_target,test_size=0.2,random_state=42)

#Scaled features
scaler_x = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)



#Model 
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(x_train,y_train)

#Predict 
y_pred = logistic_model.predict(x_test)
print("\n Accuracy Score\n:", accuracy_score(y_test,y_pred))
print("\n Precision Score\n:", precision_score(y_test,y_pred))
print("\n Recall\n:", recall_score(y_test,y_pred))
print("\n F1 Score\n:", f1_score(y_test,y_pred))

#Plotting 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logistic_model,x_test,y_test,ax=axes[0],cmap='Blues')
axes[0].set_title("Confusion Matrix")

#ROC Curve
RocCurveDisplay.from_estimator(logistic_model,x_test,y_test,ax=axes[1])
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.show()
"""
#Actvity 2 using Position_Slaraies

#In this activity we will use only feature X Level as other feature when used
#for polynomial regression so Degree 3 and 4  will cause overfiiting so to avoid it we use one feature x and
#compare with output y (Salary) using polunomail degree 2 , 3 and 4 
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , PolynomialFeatures

data_slaraies = pd.read_csv('Position_Salaries.csv')
print(data_slaraies.head())
print(data_slaraies.columns)

#Features and Target data
X_feature = data_slaraies[['Level']].values
Y_Target = data_slaraies['Salary'].values

#Train Test Split
x_train,x_test,y_train,y_test = train_test_split(X_feature,Y_Target,test_size=0.2,random_state=42)

#Scale Numeric Features
standard_scalar = StandardScaler()
x_train_scaled = standard_scalar.fit_transform(x_train)
x_test_scaled = standard_scalar.transform(x_test)

#Linear Regression Model
L_model = LinearRegression()
L_model.fit(x_train_scaled,y_train)
y_predict_linear = L_model.predict(x_test_scaled)

# Evaluation 
RMSE_error = np.sqrt(mean_squared_error(y_test,y_predict_linear))
R2_error = r2_score(y_test,y_predict_linear)

print("\n--- Linear Regression ---\n")
print("\nRMSE:\n", RMSE_error)
print("\nR2:\n"  , R2_error)

## Linear Regression Polynomail degree 2
poly_2 = PolynomialFeatures(degree=2)
x_train_2= poly_2.fit_transform(x_train_scaled)
x_test_2=poly_2.transform(x_test_scaled)

Le_model2 = LinearRegression()
Le_model2.fit(x_train_2,y_train)
y_predict_linear2 = Le_model2.predict(x_test_2)

#Error
RMSE_error_2 = np.sqrt(mean_squared_error(y_test,y_predict_linear2))
R2_error_2 = r2_score(y_test,y_predict_linear2)

print("\n--- Polynomial Degree 2 ---\n")
print("\nRMSE-2:\n", RMSE_error_2)
print("\nR2-2:\n"  , R2_error_2)

#Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_feature, Y_Target, color="blue", label="Actual Data")

X_grid_2 = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
X_grid_2_scaled = standard_scalar.transform(X_grid_2)
X_grid_2_poly = poly_2.transform(X_grid_2_scaled)
plt.plot(X_grid_2,Le_model2.predict(X_grid_2_poly),color="red",label="Polynomial Fit degree = 2",)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression Degree 2")
plt.legend()
plt.show()


##### Linear Regression Using Polynomial Degree 3
poly_3 = PolynomialFeatures(degree=3)
x_train_3= poly_3.fit_transform(x_train_scaled)
x_test_3=poly_3.transform(x_test_scaled)

 
Le_model3 = LinearRegression()
Le_model3.fit(x_train_3,y_train)
y_predict_linear3 = Le_model3.predict(x_test_3)
#Error
RMSE_error_3 = np.sqrt(mean_squared_error(y_test,y_predict_linear3))
R2_error_3 = r2_score(y_test,y_predict_linear3)

print("\n--- Polynomial Degree 3 ---\n")
print("\nRMSE_3:\n", RMSE_error_3)
print("\nR2_3:\n"  , R2_error_3)
#Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_feature, Y_Target, color="blue", label="Actual Data")


X_grid_3 = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
X_grid_3_scaled = standard_scalar.transform(X_grid_3)
X_grid_3_poly = poly_3.transform(X_grid_3_scaled)
plt.plot(X_grid_3,Le_model3.predict(X_grid_3_poly),color="red",label="Polynomial Fit degree = 3",)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression Degree 3")
plt.legend()
plt.show()


#### Linear Regression using Polynomial degree 4
poly_4 = PolynomialFeatures(degree=4)
x_train_4= poly_4.fit_transform(x_train_scaled)
x_test_4=poly_4.transform(x_test_scaled)

#Linear Regression Model using polynomial degree 2
Le_model4 = LinearRegression()
Le_model4.fit(x_train_4,y_train)
y_predict_linear4 = Le_model4.predict(x_test_4)
#Error
RMSE_error_4 = np.sqrt(mean_squared_error(y_test,y_predict_linear4))
R2_error_4 = r2_score(y_test,y_predict_linear4)

print("--- Linear Regression ---")
print("\nRMSE:\n", RMSE_error_4)
print("\nR2:\n"  , R2_error_4)
#Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_feature, Y_Target, color="blue", label="Actual Data")


X_grid_4 = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
X_grid_4_scaled = standard_scalar.transform(X_grid_4)
X_grid_4_poly = poly_4.transform(X_grid_4_scaled)
plt.plot(X_grid_4,Le_model4.predict(X_grid_4_poly),color="red",label="Polynomial Fit degree = 4",)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression Degree 4")
plt.legend()
plt.show()

#Model Perfomrnace Comparison
results_comparison = pd.DataFrame({'Models': ['Linear Regression', 'Poly Degree 2', 'Poly Degree 3', 'Poly Degree 4'],
    'RMSE_error': [RMSE_error, RMSE_error_2, RMSE_error_3, RMSE_error_4],
    'R2 Score': [R2_error, R2_error_2, R2_error_3, R2_error_4]
})

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(results_comparison.to_string(index=False))

"""