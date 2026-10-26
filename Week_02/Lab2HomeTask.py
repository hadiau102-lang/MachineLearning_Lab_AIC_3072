#Hadia Usman #BSCS 7th A  # 23-NTU-CS-1029


#Mannual Linear Regression Code  
#Home Task 1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Data_brain = pd.read_csv('headbrain.csv')
Data_brain.head()

# Linear Regression Selecting and Computing X and Y features 
#Extract values using . values
X_data = Data_brain['Head Size(cm^3)'].values
y_Target_data = Data_brain['Brain Weight(grams)'].values

#Calculatig its parametrs understanding behind the work equation how to
#calulate the Weight or say w1 (slope which tell rate of chnage in input paarmeter)
#and w0 (bias which is basically starting point )

x_sums = np.sum(X_data)
x_squred_sum = np.sum(X_data*X_data)
n_points = len(X_data)

print("\nSum of X:\n",x_sums)
print("\nSum of  Squared X:\n",x_squred_sum)
print("\nLength of X:\n",n_points)

#Now calculation cross deviation and deviation of x and finding the value of 
#w1 (slope)
numenator = n_points*np.sum(X_data * y_Target_data)- np.sum(X_data)*np.sum(y_Target_data)
denomenator = n_points*np.sum(X_data * X_data)- (np.sum(X_data))**2
#Now finding w1 (slope)
w1_slope = numenator/denomenator

#Now calculating W0 bias (base value / y-intercept)
w0_bias = (np.sum(y_Target_data)-w1_slope*(np.sum(X_data)))/n_points

print("\nValue of Slope:\n",w1_slope)
print("\nValue of bias:\n",w0_bias)

#Plotting it
maximum_x_value = np.max(X_data)
minimum_x_value = np.min(X_data)

#Calculating the lines value of x and y now 
x1_data= np.linspace(minimum_x_value,maximum_x_value)
y1_data=w0_bias + w1_slope * x1_data
#Plotting it
plt.plot(x1_data,y1_data,color='red',label='Regression Line')

#plotting it
plt.scatter(X_data,y_Target_data,c='green',label='Scatter Plot')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#Calculation RMSE TO see how much actual and predict value differnence
rmse_error = 0
for i in range(n_points):
    y_prediction = w0_bias + w1_slope * X_data[i]
    rmse_error = rmse_error + (y_Target_data[i] - y_prediction) ** 2
rmse_error = np.sqrt(rmse_error/n_points)
print("\nRMSE_Error:\n",rmse_error)

#R2_SCORE used to see how much the model has understand he underlying pattern of
# the trainning data
ss_tot = 0
ss_res = 0
y_mean_data = np.mean(y_Target_data)
for i in range(n_points):
    Y_pred = w0_bias + w1_slope * X_data[i]
    ss_tot += (y_Target_data[i] - y_mean_data) ** 2 
    ss_res += (y_Target_data[i] - Y_pred) ** 2 
r2_score = 1 - (ss_res/ss_tot)
print("\nR2 Score:\n",r2_score)
"""

#Mannual Logistic Regression Code 
#Home Task 2
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#Load data set
data_social = pd.read_csv('Social_Network_Ads.csv')

#Preprocessing
Labelen = LabelEncoder()
data_social['Gender']=Labelen.fit_transform(data_social['Gender'])

#Feature and Target Values
X_data_feature = data_social[['Gender', 'Age', 'EstimatedSalary']].values 
y_target_data = data_social['Purchased'].values 

#Fit daa set into training ad testing one
x_train , x_test , y_Train , y_test = train_test_split(X_data_feature,y_target_data,test_size=0.2,random_state=42)

#Scale features 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Logistic regression mannual implmentation 

class LogisticRegression:
  def __init__(self, learning_rate=0.01, num_iterations=1000):
   self.learningrate = learning_rate
   self.iterations = num_iterations
   self.weights = None
   self.bias = None

#Sigmoid convert it into valid prob. between 0 and 1
  def sigmoid(self, z):
   return 1 / (1 + np.exp(-z))
#Fit it 
  def fit(self, X, y):
   num_samples, num_features = X.shape
   self.weights = np.zeros(num_features)
   self.bias = 0
   #ITEARTION
   for i in range(self.iterations):
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = self.sigmoid(linear_model)
 # Update weights and bias by calculating Gradient Gradient
      dw_update = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
      db_update= (1 / num_samples) * np.sum(y_predicted - y)
      self.weights -= self.learningrate * dw_update
      self.bias -= self.learningrate * db_update

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    return [1 if i > 0.5 else 0 for i in self.sigmoid(linear_model)]

#Train model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(x_train, y_Train)

#mODEL PREDICTION AND ACCURACY 
predictions = model.predict(x_test)
print("Accuracy of Model:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
"""

##########   ACTIVITIES ############
#Activity 1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data_medical = pd.read_csv('Medical Cost Personal Datasets.csv')
print(data_medical.head())


#LEncoding using label Encoder
Labelencoder = LabelEncoder()
data_medical['smoker'] = Labelencoder.fit_transform(data_medical['smoker'])
data_medical['region'] = Labelencoder.fit_transform(data_medical['region'])


print(data_medical['smoker'])
print(data_medical['region'])

#Feature and Target 

X_feature_data = data_medical[['age','bmi', 'children','smoker','region']]
y_target_data = data_medical['charges']


#Training dataset
x_train,x_test,y_train,y_test = train_test_split(X_feature_data,y_target_data,test_size=0.2,random_state=42)


#Standard Scaling
standard_scaler = StandardScaler()
x_f = ['age','bmi','children']

x_train[x_f]=standard_scaler.fit_transform(x_train[x_f])
x_test[x_f]=standard_scaler.transform(x_test[x_f])

Le_model = LinearRegression()
Le_model.fit(x_train,y_train)


#Evaluate using RMSE and R2  score know

y_prediction = Le_model.predict(x_test)

RMSE_error = np.sqrt(mean_squared_error(y_test,y_prediction))
R2_score = r2_score(y_test,y_prediction)
print("\nRMSE:\n",RMSE_error)
print("R2_score:",R2_score)

#Plotting
plt.scatter(y_test,y_prediction,color='green' , alpha=0.6, edgecolors="k")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="r",linestyle="--",lw=2,label="Ideal Fit")
plt.xlabel('Actual Costs')
plt.ylabel('Predicted Costs')
plt.title('Actual vs Predicted Medical Insurance Costs')
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.show()
"""
#Activity 2
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
data_telco = pd.read_csv('Telco Customer Churn.csv')
print(data_telco.head())
print(data_telco.columns)

#Encoding Label and One He Encoding

Labelencoder = LabelEncoder()
y_target = Labelencoder.fit_transform(data_telco['Churn'])

X_features = data_telco[['tenure','InternetService','Contract','MonthlyCharges']]
X_f = pd.get_dummies(X_features,columns=['InternetService','Contract'],drop_first=True)

#Train test 
x_train , x_test , y_train, y_test = train_test_split(X_f,y_target,test_size=0.2,random_state=42)

#Scaled features
scaler_x = StandardScaler()
x_cols = ['tenure','MonthlyCharges']
x_train[x_cols] = scaler_x.fit_transform(x_train[x_cols])
x_test[x_cols] = scaler_x.transform(x_test[x_cols])



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
#Activity 3
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , PolynomialFeatures

data_car_price = pd.read_csv('Car Price Prediction.csv')
print(data_car_price.head())
print(data_car_price.columns)

#Features and Target data
X_feature = data_car_price[['enginesize','curbweight','horsepower','citympg']]
Y_Target = data_car_price['price']

#Train Test Split
x_train,x_test,y_train,y_test = train_test_split(X_feature,Y_Target,test_size=0.2,random_state=42)

#Scale Numeric Features
std_scalar = StandardScaler()
x_train_scaled = std_scalar.fit_transform(x_train)
x_test_scaled = std_scalar.transform(x_test)

#Linear Regression Model
Le_model = LinearRegression()
Le_model.fit(x_train_scaled,y_train)
y_predict_linear = Le_model.predict(x_test_scaled)

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

RMSE_error_2 = np.sqrt(mean_squared_error(y_test,y_predict_linear2))
R2_error_2 = r2_score(y_test,y_predict_linear2)

print("\n--- Polynomial Degree 2 ---\n")
print("\nRMSE-2:\n", RMSE_error_2)
print("\nR2-2:\n"  , R2_error_2)

##### Linear Regression Using Polynomial Degree 3
poly_3 = PolynomialFeatures(degree=3)
x_train_3= poly_3.fit_transform(x_train_scaled)
x_test_3=poly_3.transform(x_test_scaled)

 
Le_model3 = LinearRegression()
Le_model3.fit(x_train_3,y_train)
y_predict_linear3 = Le_model3.predict(x_test_3)

RMSE_error_3 = np.sqrt(mean_squared_error(y_test,y_predict_linear3))
R2_error_3 = r2_score(y_test,y_predict_linear3)

print("\n--- Polynomial Degree 3 ---\n")
print("\nRMSE_3:\n", RMSE_error_3)
print("\nR2_3:\n"  , R2_error_3)

#### Linear Regression using Polynomial degree 4
poly_4 = PolynomialFeatures(degree=4)
x_train_4= poly_4.fit_transform(x_train_scaled)
x_test_4=poly_4.transform(x_test_scaled)

#Linear Regression Model using polynomial degree 2
Le_model4 = LinearRegression()
Le_model4.fit(x_train_4,y_train)
y_predict_linear4 = Le_model4.predict(x_test_4)

RMSE_error_4 = np.sqrt(mean_squared_error(y_test,y_predict_linear4))
R2_error_4 = r2_score(y_test,y_predict_linear4)

print("--- Linear Regression ---")
print("\nRMSE:\n", RMSE_error_4)
print("\nR2:\n"  , R2_error_4)


#Model Perfomrnace Comparison
results_comparison = pd.DataFrame({'Models': ['Linear Regression', 'Poly Degree 2', 'Poly Degree 3', 'Poly Degree 4'],
    'RMSE_error': [RMSE_error, RMSE_error_2, RMSE_error_3, RMSE_error_4],
    'R2 Score': [R2_error, R2_error_2, R2_error_3, R2_error_4]
})

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(results_comparison.to_string(index=False))

#Plotting 

#Createing a 2x2 gird layout so one can see the differnces between each 
#using sorting to sort our y_test as when plottng for polynomail regression cannot plot
#with muliple x features so we use it and sort the y_test values to see our result

sort_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sort_idx]

models = [
    ('Linear Regression', y_predict_linear[sort_idx], 'blue'),
    ('Polynomial Degree 2', y_predict_linear2[sort_idx], 'gold'),
    ('Polynomial Degree 3', y_predict_linear3[sort_idx], 'green'),
    ('Polynomial Degree 4', y_predict_linear4[sort_idx], 'orange'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (title, y_pred_sorted, color) in zip(axes.flatten(), models):
  # Actual Data Points
  ax.scatter(range(len(y_test_sorted)),y_test_sorted,color='red',label='Actual Data Points', alpha=0.7,zorder=3,)

  #  Fitted Line/Curves for Linear Regression and Polynomial DEgree 2,3,4
  ax.plot(range(len(y_test_sorted)),y_pred_sorted,color=color,linewidth=2,label=f'Fitted Line ({title})',)

  ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
  ax.set_xlabel('Actual Car Price')
  ax.set_ylabel('Predicted Car Price')
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend(loc='upper left')

plt.tight_layout(h_pad=3.0, w_pad=2.0)
plt.show()
"""