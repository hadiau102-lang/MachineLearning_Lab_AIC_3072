#Hadia Usman 
#23-NTU-CS-1029
#BSCS 7th A 
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

#K- values
k_values = [1,3,5,7,9]

#Train a Knn Classifier for each value of k and record  the accuracy score 
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy=knn.score(x_test,y_test)
    accuracy_scores.append(accuracy)

#PLolt it
plt.scatter(k_values,accuracy_scores)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k values")
plt.show()