#Hadia Usman , 23-NTU-CS-1029 , BSCS 7th A 
import numpy as np
import pandas as pd 
import seaborn as sns 
import sklearn 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xg

elec_data = fetch_openml(name='electricity',version=1)
elec_df = elec_data.frame
print("Data Shape:",elec_data.data.shape)
print("Feature Name:",elec_data.feature_names)
print("Target Name:",elec_data.target_names)

train_data , test_data = train_test_split(elec_df,test_size=0.25,random_state=20)
print("Tarining Sample:",len(train_data))
print("Testing  Sample:",len(test_data))


train_data.head(10)

train_subset = train_data.sample(n=5000, random_state=42)
X_train = train_subset.drop('class', axis=1)
y_train = train_subset['class']

train_data.info()
train_data.describe()

train_data.isnull().sum()
corr = train_data.corr(numeric_only=True)
plt.figure(figsize=(12,7))
sns.heatmap(corr, annot=True,cmap='crest')
plt.show()

plt.figure(figsize=(12,7))
sns.countplot(data=train_data,x='class')
plt.show()

plt.figure(figsize=(13,8))
sns.countplot(data=train_data,x='day',hue='class')
plt.show()

train_data.hist(bins=50,figsize=(15,10))
plt.show()


X_train = train_data.drop('class', axis=1)
y_train = train_data['class']


label_enc = LabelEncoder()
y_train_prepared = label_enc.fit_transform(y_train)

print(label_enc.classes_)



def accuracy(model, data, labels):
    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)

    return acc

#Voting Classifier
X_train_num = X_train.copy()
X_train_num['day'] = X_train_num['day'].astype(int)
models = [('lr', LogisticRegression(max_iter=1000)),
          ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
          ('svc', SVC(probability=True, random_state=42))]



voting_hard = VotingClassifier(models, voting='hard')
voting_soft = VotingClassifier(models, voting='soft')
voting_hard.fit(X_train_num, y_train_prepared)
voting_soft.fit(X_train_num, y_train_prepared)
print(accuracy(voting_hard, X_train_num, y_train_prepared))
print(accuracy(voting_soft, X_train_num, y_train_prepared))

#Bagging Classifier
bag_class =BaggingClassifier(DecisionTreeClassifier(class_weight='balanced'),max_samples=0.5,max_features=0.5,bootstrap=False)
bag_class.fit(X_train,y_train_prepared)
print(accuracy(bag_class,X_train,y_train_prepared))

#GradientBoosting
GRD_bag_class =GradientBoostingClassifier(n_estimators=500,learning_rate=0.8,random_state=42,max_depth=2)

GRD_bag_class.fit(X_train, y_train_prepared)
print(accuracy(GRD_bag_class, X_train, y_train_prepared))

#Adaboast Classifier
AD_boast_class =AdaBoostClassifier( estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),n_estimators=300, learning_rate=0.5)

AD_boast_class.fit(X_train, y_train_prepared)
print(accuracy(AD_boast_class, X_train, y_train_prepared))

#XGBOast 
Xg_class =xg.XGBClassifier()
Xg_class.fit(X_train_num, y_train_prepared)
print(accuracy(Xg_class, X_train_num, y_train_prepared))

#Stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
]
stack_clf = StackingClassifier(estimators=base_models,final_estimator=LogisticRegression(max_iter=1000),cv=5)
stack_clf.fit(X_train_num, y_train_prepared)
print(accuracy(stack_clf, X_train_num, y_train_prepared))

X_test = test_data.drop('class',axis=1)
Y_test=test_data['class']
y_test_prepared=label_enc.transform(Y_test)
print(accuracy(GRD_bag_class,X_test,y_test_prepared))
print(accuracy(bag_class,X_test,y_test_prepared))

X_test_num = X_test.copy()
X_test_num['day']=X_test_num['day'].astype(int)
print(accuracy(voting_hard,X_test_num,y_test_prepared))
print(accuracy(voting_soft,X_test_num,y_test_prepared))
print(accuracy(Xg_class,X_test_num,y_test_prepared))
print(accuracy(stack_clf,X_test_num,y_test_prepared))
print(accuracy(AD_boast_class,X_test_num,y_test_prepared))