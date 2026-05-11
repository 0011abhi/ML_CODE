import numpy as np
import matplotlib.pyplot as plt

# Gaussian Kernel
def kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi)**2) / (2 * tau**2))

# Locally Weighted Regression
def lwr(x, X, y, tau):

    weights = np.array([kernel(x, X[i], tau) for i in range(len(X))])

    W = np.diag(weights)

    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y

    return x @ theta

# Generate data
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)

# Add bias term
X_bias = np.c_[np.ones(X.shape), X]

# Test points
x_test = np.linspace(0, 2*np.pi, 200)
x_test_bias = np.c_[np.ones(x_test.shape), x_test]

# Predictions
tau = 0.5
y_pred = np.array([lwr(x, X_bias, y, tau) for x in x_test_bias])

# Plot
plt.scatter(X, y, label="Training Data")
plt.plot(x_test, y_pred, label="LWR Curve")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()