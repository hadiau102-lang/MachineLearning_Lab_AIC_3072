#SVC
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")

df_data = pd.read_csv('income_evaluation.csv')
df_data.head()

categorical_features = df_data.select_dtypes(include=['object']).columns
numerical_features = df_data.select_dtypes(exclude=['object']).columns

# Apply one-hot encoding to categorical features
df = pd.get_dummies(df_data, columns=categorical_features, drop_first=True)

# Separate features and target variable again after one-hot encoding
X_feature= df.drop("income_ >50K", axis=1) # Adjust target column name based on get_dummies output
y_target = df["income_ >50K"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_feature, y_target, test_size=0.2, random_state=101)

# Scale numerical features
scaler = StandardScaler()
x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test[numerical_features] = scaler.transform(x_test[numerical_features])

print(x_train.head())

svc = SVC(random_state = 101)
svc.fit(x_train,y_train)
accuracies = cross_val_score(svc,x_train,y_train,cv=5)
print("Train Score:", np.mean(accuracies))
print("Test Score:", svc.score(x_test,y_test))

