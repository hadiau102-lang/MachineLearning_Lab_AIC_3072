#Hadia Usman 
#23-NTU-CS-1029
#BSCS 7th A
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings 
warnings.filterwarnings('ignore')


water_data = pd.read_csv('water_potability.csv')

print("Data Shape:",water_data.shape)
print("\nFirst 10 rows:",water_data.head(10))
print("\nMissing Values:",water_data.isnull().sum())
print("\nClass Distribution :",water_data['Potability'].value_counts())
X_feature=water_data.drop('Potability',axis=1)
Y_target=water_data['Potability']
x_train ,x_test, y_train,y_test = train_test_split(X_feature,Y_target,test_size=0.25,random_state=20)

imputer = SimpleImputer(strategy='median')
columns_toimpute=['ph','Sulfate','Trihalomethanes']

x_train[columns_toimpute]=imputer.fit_transform(x_train[columns_toimpute])
x_test[columns_toimpute]=imputer.transform(x_test[columns_toimpute])

scaler=StandardScaler()

X_trained_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
X_train_scaled_df = pd.DataFrame(X_trained_scaled, columns=X_feature.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_feature.columns)

lr_model = LogisticRegression(random_state=42)
svc_model = SVC(probability=True,random_state=42)
rf_model = RandomForestClassifier(random_state=42)


lr_model.fit(X_train_scaled_df,y_train)
svc_model.fit(X_train_scaled_df,y_train)
rf_model.fit(x_train,y_train)


estimatros = [('lr', LogisticRegression(random_state=42)),
          ('svc', SVC(probability=True, random_state=42)),
          ('rf', RandomForestClassifier(random_state=42))]



voting_hard = VotingClassifier(estimatros, voting='hard')
voting_soft = VotingClassifier(estimatros, voting='soft')
voting_hard.fit(X_train_scaled_df, y_train)
voting_soft.fit(X_train_scaled_df, y_train)


#Bagging Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)
bag_10_class =BaggingClassifier(DecisionTreeClassifier(random_state=42),max_samples=0.5,n_estimators=10,bootstrap=True,random_state=42)
bag_100_class =BaggingClassifier(DecisionTreeClassifier(random_state=42),max_samples=0.5,n_estimators=100,bootstrap=True,random_state=42)
bag_100_F_class =BaggingClassifier(DecisionTreeClassifier(random_state=42),max_samples=0.5,n_estimators=100,bootstrap=False,random_state=42)
bag_10_class.fit(x_train,y_train)
bag_100_class.fit(x_train,y_train)
bag_100_F_class.fit(x_train,y_train)

models_dict = {
    "Logistic Regression": (lr_model, X_test_scaled_df),
    "SVC": (svc_model, X_test_scaled_df),
    "Random Forest": (rf_model, x_test),
    "Hard Voting": (voting_hard, X_test_scaled_df),
    "Soft Voting": (voting_soft, X_test_scaled_df),
    "Single Decision Tree": (dt_model, x_test),
    "Bagging (10, Boot)": (bag_10_class, x_test),
    "Bagging (100, Boot)": (bag_100_class, x_test),
    "Pasting (100, No Boot)": (bag_100_F_class, x_test)
}

results = []
for name, (model, data) in models_dict.items():
    preds = model.predict(data)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0)
    })

results_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(results_df)

# Confusion matrix of the best model (e.g., Random Forest or Soft Voting)
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")
best_model, best_data = models_dict[best_model_name]
best_preds = best_model.predict(best_data)

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()