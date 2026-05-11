import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

plt.figure(figsize=(12,8))
plot_tree(model, filled=True)
plt.show()

sample = X[0].reshape(1,-1)
prediction = model.predict(sample)

print("Predicted Class:", data.target_names[prediction[0]])
