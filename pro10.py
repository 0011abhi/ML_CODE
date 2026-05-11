import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("K-Means Clustering")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# Compare clusters with actual labels
df = pd.DataFrame({'Cluster': clusters, 'Actual': y})
print(pd.crosstab(df['Cluster'], df['Actual']))