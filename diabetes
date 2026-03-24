import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes


# Load dataset
dataset = load_diabetes(as_frame=True)
df = pd.DataFrame(dataset.data)

# Use BMI as feature
X = df[["bmi"]]
y = dataset.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)


# Linear Regression from scratch
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.weights = np.random.uniform(-1, 1, n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # Cost (MSE)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


# Train model
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# Plot actual vs predicted
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted")
plt.title("Linear Regression")
plt.legend()
plt.show()


# Plot cost convergence
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.cost_history)), model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Iteration (Training Convergence)")
plt.grid(True)
plt.show()
