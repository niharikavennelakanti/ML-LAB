# -------------------------------------------------------------
# LAB 08 – Emotion Detection using Brain Activity Dataset
# -------------------------------------------------------------
# Covers: A1–A12 + Optional O1/O2 partial setup
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------------------------------------------------
# A1. Basic Perceptron Building Blocks
# -------------------------------------------------------------

def summation_unit(inputs, weights, bias):
    """Summation unit: Computes weighted sum"""
    return np.dot(inputs, weights) + bias

# Activation Functions
def step_activation(x):
    return np.where(x >= 0, 1, 0)

def bipolar_step_activation(x):
    return np.where(x >= 0, 1, -1)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

def leaky_relu_activation(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def comparator_error(y_true, y_pred):
    """Compute sum of squared error"""
    return np.sum((y_true - y_pred) ** 2) / 2

# -------------------------------------------------------------
# A2. Perceptron Training Function
# -------------------------------------------------------------
def perceptron_train(X, y, lr=0.05, epochs=1000, activation_func=step_activation, tol=0.002):
    """Train a perceptron using custom activation"""
    np.random.seed(42)
    weights = np.random.randn(X.shape[1])
    bias = 0.1
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            net_input = summation_unit(X[i], weights, bias)
            output = activation_func(net_input)
            error = y[i] - output
            weights += lr * error * X[i]
            bias += lr * error
            total_error += error**2
        errors.append(total_error)

        if total_error <= tol:
            print(f"✅ Converged at epoch {epoch}")
            break

    return weights, bias, errors

# -------------------------------------------------------------
# A2. AND Gate Example
# -------------------------------------------------------------
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

weights, bias, errors = perceptron_train(X_and, y_and, lr=0.05, activation_func=step_activation)
print(f"Final Weights: {weights}, Bias: {bias}")

plt.plot(errors)
plt.title("AND Gate Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Error")
plt.show()

# -------------------------------------------------------------
# A3. Compare Activation Functions for AND Gate
# -------------------------------------------------------------
activations = {
    "Bipolar Step": bipolar_step_activation,
    "Sigmoid": sigmoid_activation,
    "ReLU": relu_activation
}

convergence = {}
for name, act in activations.items():
    _, _, err = perceptron_train(X_and, y_and, activation_func=act)
    convergence[name] = len(err)

plt.bar(convergence.keys(), convergence.values(), color='orange')
plt.title("Comparison of Convergence (AND Gate)")
plt.ylabel("Epochs to Converge")
plt.show()

# -------------------------------------------------------------
# A4. Learning Rate Variation
# -------------------------------------------------------------
rates = np.arange(0.1, 1.1, 0.1)
epochs_needed = []

for lr in rates:
    _, _, err = perceptron_train(X_and, y_and, lr=lr)
    epochs_needed.append(len(err))

plt.plot(rates, epochs_needed, marker='o')
plt.title("Learning Rate vs Epochs to Converge")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.show()

# -------------------------------------------------------------
# A5. XOR Gate (Non-linearly separable)
# -------------------------------------------------------------
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

weights, bias, errors = perceptron_train(X_xor, y_xor, activation_func=sigmoid_activation)
plt.plot(errors)
plt.title("XOR Gate (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# -------------------------------------------------------------
# A6. Customer Data Example
# -------------------------------------------------------------
customers = pd.DataFrame({
    "Candies": [20,16,27,19,24,22,15,18,21,16],
    "Mangoes": [6,3,6,1,4,1,4,4,1,2],
    "Milk": [2,6,2,2,2,5,2,2,4,4],
    "Payment": [386,289,393,110,280,167,271,274,148,198],
    "HighValue": ["Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No"]
})
customers["HighValue"] = customers["HighValue"].map({"Yes":1,"No":0})

X_cust = customers[["Candies","Mangoes","Milk","Payment"]].values
y_cust = customers["HighValue"].values

weights, bias, errors = perceptron_train(X_cust, y_cust, activation_func=sigmoid_activation)
plt.plot(errors)
plt.title("Customer Data (Sigmoid Activation)")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# -------------------------------------------------------------
# A11. MLPClassifier() for Logic Gates
# -------------------------------------------------------------
from sklearn.neural_network import MLPClassifier

# AND Gate using MLP
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
mlp_and.fit(X_and, y_and)
print("\nMLP AND Gate Results:")
print("Predictions:", mlp_and.predict(X_and))

# XOR Gate using MLP
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.05, max_iter=1000)
mlp_xor.fit(X_xor, y_xor)
print("\nMLP XOR Gate Results:")
print("Predictions:", mlp_xor.predict(X_xor))

# -------------------------------------------------------------
# A12. MLP on Project Dataset (data set 2.csv)
# -------------------------------------------------------------
data = pd.read_csv("data set 2.csv")
target_col = 'label' if 'label' in data.columns else data.columns[-1]

X = data.drop(columns=[target_col])
y = data[target_col]
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlp_project = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', solver='adam',
                            learning_rate_init=0.001, max_iter=500)
mlp_project.fit(X_train, y_train)

y_pred = mlp_project.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ MLPClassifier on Project Dataset - Accuracy: {acc:.3f}")

# Plot learning curve
plt.plot(mlp_project.loss_curve_)
plt.title("Project Dataset - MLP Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
