import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Plot Tree
plt.figure(figsize=(12,8))
plot_tree(model, filled=True)
plt.show()

# Predict new sample
sample = X[0].reshape(1,-1)
prediction = model.predict(sample)

print("Predicted Class:", data.target_names[prediction[0]])