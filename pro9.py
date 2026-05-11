import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = fetch_olivetti_faces()

# Features and labels
X = data.images.reshape((400, -1))
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Show sample images
fig, axes = plt.subplots(2,5, figsize=(10,5))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(64,64), cmap='gray')
    ax.set_title(f"P:{y_pred[i]} A:{y_test[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()