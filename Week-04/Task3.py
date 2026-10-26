#Hadia Usman , 23-NTU-CS-1029 , BSCS 7thA
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#Load wind dataset 
wine_data = load_wine()
X_wine_features , Y_target = wine_data.data , wine_data.target

#Train Test Split
x_train , x_test , y_train, y_test = train_test_split(X_wine_features,Y_target,test_size=0.2,random_state=42)

#p- values for knn
p_values = [1,2,3,4,5]

#Train a Knn Classifier for each value of k and record  the accuracy score 
accuracy_scores = []
for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=5,p=p)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy=knn.score(x_test,y_test)
    accuracy_scores.append(accuracy)

print("Print Accuracy score for each value of p:")
for p,accuracy in zip(p_values,accuracy_scores):
    print(f"p={p}:{accuracy}")
print()

#PLolt it
plt.scatter(p_values,accuracy_scores)
plt.xlabel("p")
plt.ylabel("Accuracy")
plt.title("Accuracy vs p values")
plt.show()