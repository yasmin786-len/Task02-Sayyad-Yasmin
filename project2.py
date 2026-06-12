

# Import libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

import pandas as pd

#  DATASET loading

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Loaded Successfully")
print("Total Samples:", len(X))



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

#  FEATURE SCALING


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData Scaling Completed")

#  BUILD KNN MODEL


model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed")

# PREDICTION


predictions = model.predict(X_test)

# ACCURACY


accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# CONFUSION MATRIX


cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

#  F1 SCORE

f1 = f1_score(
    y_test,
    predictions,
    average='weighted'
)

print("\nF1 Score:")
print(round(f1, 4))

# CLASSIFICATION REPORT

print("\nClassification Report:")
print(classification_report(y_test, predictions))

#TEST CUSTOM FLOWER

print("\n------ Custom Prediction ------")

sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

new_flower = scaler.transform(new_flower)

result = model.predict(new_flower)

flower_name = iris.target_names[result[0]]

print("\nPredicted Flower:", flower_name)
