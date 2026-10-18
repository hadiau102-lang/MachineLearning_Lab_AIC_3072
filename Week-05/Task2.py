#voting classifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")

# 1. Load the Sklearn Digits Dataset (8x8 pixel images of handwritten digits)
digits = load_digits()
X = digits.data
y = digits.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 'some_digit' using an example sample from your test data
# Let's grab the first sample image from X_test to predict later
some_digit = X_test[0] 

# 3. Initialize Base Classifiers
log_clf = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC()

# 4. Create and Train the Voting Classifier Ensemble
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], 
    voting='hard'
)

voting_clf.fit(X_train, y_train)

# 5. Predict on a single sample (reshaped to a 2D array to avoid scikit-learn format warnings)
predicted_digit = voting_clf.predict([some_digit])
print(f"Prediction for some_digit: {predicted_digit[0]}")
print(f"Actual label for some_digit: {y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]}\n")

# 6. Evaluate and Compare All Classifiers
print("Individual and Ensemble Accuracy Scores:")
print("-" * 40)
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__:<25} Matrix Accuracy: {accuracy_score(y_test, y_pred):.4f}")
