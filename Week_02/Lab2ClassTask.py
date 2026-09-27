#Hadia Usman
#23-NTU-CS-1029
#BSCS-7th A

#Multivarieant LR

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

from sklearn.model_selection import train_test_split

df_data= pd.read_csv("Admission_Predict.csv")
columns = df_data.columns
df_data.drop("Serial No.",axis=1,inplace=True)
print(df_data.columns)
y = df_data['Chance of Admit ']
df_data.drop("Chance of Admit ",axis=1,inplace=True)
df_data.head()

linear_regression = LinearRegression()
linear_regression.fit(df_data[['GRE Score']], y)
plt.scatter(df_data['GRE Score'], y, color='green', label='Actual Data', alpha=0.5)
plt.plot(df_data['GRE Score'], linear_regression.predict(df_data[['GRE Score']]), color='red', linewidth=3, label='Regression Line')
plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()

x_train_data, x_test, y_train_data, y_test = train_test_split(df_data,y,test_size=0.2,random_state=42)
lr1 = LinearRegression()
lr1.fit(x_train_data, y_train_data)
pred = lr1.predict(x_test)

rmse_error = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("RMSE:", rmse_error)

coefficients = lr1.coef_
features = df_data.columns
plt.figure(figsize=(8, 5))
plt.barh(features,coefficients,color="teal")

plt.xlabel("Coefficient Value (Weight)")
plt.title("Feature Importance in Multivariate Linear Regression")
plt.axvline(x=0,color="black",linewidth=0.8)

plt.show()
"""
#Logistic Regression

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('Social_Network_Ads.csv')
le = LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])

x= data[['Gender','Age','EstimatedSalary']].values
y = data['Purchased'].values

X_train , X_test , Y_Train , Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scalar = StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

model = LogisticRegression()
model.fit(X_train,Y_Train)
y_pred=model.predict(X_test)

print("Accuracy :",accuracy_score(Y_test, y_pred))
print("Confusion Matrix :\n",confusion_matrix(Y_test, y_pred))
print("Classification Report  :\n",classification_report(Y_test, y_pred))
"""
#Polynomial_Regression

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_1 = pd.read_csv('Position_Salaries.csv')
x=data_1[['Level']].values
y=data_1[['Salary']].values
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(x)

poly_model = LinearRegression()
poly_model.fit(X_poly,y)

y_pred = poly_model.predict(X_poly)

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
"""