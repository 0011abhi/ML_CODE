import pandas as pd

# Load dataset
df = pd.read_csv("training_data.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Find-S Algorithm
hypothesis = None

for i in range(len(X)):
    
    if y[i].lower() == "yes":

        if hypothesis is None:
            hypothesis = X.iloc[i].tolist()

        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != X.iloc[i, j]:
                    hypothesis[j] = "?"

print("Final Hypothesis:", hypothesis)