import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Select numeric columns
columns = df.select_dtypes(include=['float64', 'int64']).columns

# Histograms
df[columns].hist(figsize=(15,10), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots
df[columns].plot(kind='box', subplots=True, layout=(3,3), figsize=(15,10))
plt.tight_layout()
plt.show()

# Outlier Detection using IQR
print("Outliers Detection:")
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col}: {len(outliers)} outliers")