
#Mannual Logistic Regression Code and Poynomail Regression One
#Logistic Regression
#Class Task 2
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Load data set
data_social_1 = pd.read_csv('Social_Network_Ads.csv')

#Preprocessing
Labelenc_1 = LabelEncoder()
data_social_1['Gender']=Labelenc_1.fit_transform(data_social_1['Gender'])

#Feature and Target Values
X_data = data_social_1[['Gender', 'Age', 'EstimatedSalary']].values 
y_target = data_social_1['Purchased'].values 

#Fit daa set into training ad testing one
x_train1 , x_test1 , y_Train1 , y_test1 = train_test_split(X_data,y_target,test_size=0.2,random_state=42)

#Scale features  so that we can have standard scaler 
std_scaler = StandardScaler()
x_train1 = std_scaler.fit_transform(x_train1)
x_test1 = std_scaler.transform(x_test1)

#Logistic regression mannual implmentation for Binary Classifcation like 0 or 1

class LogisticRegression:
  def __init__(self, learning_rate=0.01, num_iterations=1000):
   self.learningrate = learning_rate
   self.iterations = num_iterations
   self.weights = None
   self.bias = None

#Sigmoid convert it into valid prob. between 0 and 1
  def sigmoid(self, z):
   return 1 / (1 + np.exp(-z))
#Fit it means Training the data 
  def fit(self, X, y):
   num_samples, num_features = X.shape
   self.weights = np.zeros(num_features)
   self.bias = 0
   #ITEARTION
   for i in range(self.iterations):
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = self.sigmoid(linear_model)
 # Update weights and bias by calculating Gradient Descent
      dw_update = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
      db_update= (1 / num_samples) * np.sum(y_predicted - y)
      self.weights -= self.learningrate * dw_update
      self.bias -= self.learningrate * db_update
#Predict to check the errors 
  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    return [1 if i > 0.5 else 0 for i in self.sigmoid(linear_model)]

#Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(x_train1, y_Train1)

#MODEL PREDICTION AND ACCURACY 
predictions = model.predict(x_test1)
print("Accuracy of Model:", accuracy_score(y_test1, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test1, predictions))
print("Classification Report:\n", classification_report(y_test1, predictions))
"""


#Class Task 2
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data_salaries = pd.read_csv('Position_Salaries.csv')
X_F_data = data_salaries[['Level']].values
Y_t_data = data_salaries['Salary'].values

poly_f = PolynomialFeatures(degree=4)
X_poly_f = poly_f.fit_transform(X_F_data)

poly_lin_model = LinearRegression()
poly_lin_model.fit(X_poly_f,Y_t_data)

y_pred = poly_lin_model.predict(X_poly_f)

plt.scatter(X_F_data,Y_t_data,color='blue',label='Actual Data')
plt.plot(X_F_data,y_pred,color='red',label='Polynomial Degree(4)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

r2_error =r2_score(Y_t_data,y_pred)
print("Polynomial Regression:")
print(f"R2 Score:{r2_error:4f}")
"""