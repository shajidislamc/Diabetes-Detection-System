import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import json


# Load dataset
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier()
}

# collect accuracies for each model
accuracies = {}

# Initialize variables to track the best model
best_model_name = None
best_accuracy = 0
best_model = None

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    accuracies[model_name] = float(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model
        
print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}")

# Save the best model
joblib.dump(best_model, "model/diabetes_model.joblib")
print("Model saved to model/diabetes_model.joblib")

# Compute evaluation metrics for the best model on the test set
y_best_pred = best_model.predict(X_test)
metrics = {
    "accuracy": float(accuracy_score(y_test, y_best_pred)),
    "precision": float(precision_score(y_test, y_best_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_best_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_best_pred, zero_division=0))
}
with open("model/diabetes_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to model/diabetes_metrics.json")
