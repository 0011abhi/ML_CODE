import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
X = np.random.rand(100,1)

y = np.where(X <= 0.5, 1, 2).ravel()

X_train, X_test = X[:50], X[50:]
y_train, y_test = y[:50], y[50:]

k_values = [1,3,5,20]

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(f"k={k} Accuracy =", knn.score(X_test, y_test))

    plt.scatter(X_test, y_test, label="True")
    plt.scatter(X_test, y_pred, marker='x', label="Predicted")
    plt.title(f"KNN with k={k}")
    plt.legend()
    plt.show()
