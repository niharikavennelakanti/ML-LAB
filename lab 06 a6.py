# --------------------------------------------
# LAB 06 – Emotion Detection using Brain Activity Dataset
# Tasks: Entropy, Gini Index, Information Gain, Decision Tree
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data set 2.csv")
print("Dataset Loaded Successfully ✅")
print(data.head())

# --------------------------------------------
# A1. Function to calculate Entropy
# --------------------------------------------
def calculate_entropy(y):
    """Calculate entropy for a label column."""
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Example usage
entropy_val = calculate_entropy(data.iloc[:, -1])
print(f"Entropy of target column: {entropy_val:.4f}")

# --------------------------------------------
# Equal Width Binning Function (for continuous features)
# --------------------------------------------
def equal_width_binning(column, bins=4):
    """Convert continuous data into categorical using equal width binning."""
    categories = pd.cut(column, bins=bins, labels=[f'Bin{i+1}' for i in range(bins)])
    return categories

# Example: Apply binning to numeric columns
for col in data.select_dtypes(include=np.number).columns:
    data[col] = equal_width_binning(data[col], bins=4)

# Encode target if categorical
if data.iloc[:, -1].dtype == 'object':
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])

# --------------------------------------------
# A2. Calculate Gini Index
# --------------------------------------------
def calculate_gini(y):
    """Calculate Gini index for a label column."""
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities ** 2)
    return gini

gini_val = calculate_gini(data.iloc[:, -1])
print(f"Gini Index of target column: {gini_val:.4f}")

# --------------------------------------------
# A3 & A4. Information Gain & Root Node Selection
# --------------------------------------------
def information_gain(df, feature, target):
    """Compute Information Gain for a given feature."""
    total_entropy = calculate_entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * calculate_entropy(df[df[feature] == values[i]][target])
        for i in range(len(values))
    ])
    info_gain = total_entropy - weighted_entropy
    return info_gain

target = data.columns[-1]
info_gains = {col: information_gain(data, col, target) for col in data.columns[:-1]}

root_feature = max(info_gains, key=info_gains.get)
print("\nInformation Gain for each feature:")
for feature, ig in info_gains.items():
    print(f"{feature}: {ig:.4f}")
print(f"\n📍 Root Node based on Information Gain: {root_feature}")

# --------------------------------------------
# A5. Build Decision Tree using Scikit-Learn
# --------------------------------------------
X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# --------------------------------------------
# A6. Visualize the Decision Tree
# --------------------------------------------
plt.figure(figsize=(15, 8))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=True, rounded=True)
plt.title("Decision Tree for Emotion Detection")
plt.show()

# --------------------------------------------
# A7. Decision Boundary Visualization (2 features)
# --------------------------------------------
from matplotlib.colors import ListedColormap

# Select 2 features
features = list(X.columns[:2])
X2 = X[features].apply(LabelEncoder().fit_transform)
y2 = y

# Train a simple Decision Tree on 2D features
model_2D = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_2D.fit(X2, y2)

# Plot decision boundary
x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model_2D.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
sns.scatterplot(x=X2.iloc[:, 0], y=X2.iloc[:, 1], hue=y2, palette='deep', s=60)
plt.title("Decision Boundary of Decision Tree (2D Feature Space)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()

print("\n✅ Lab 06 Execution Completed Successfully!")
