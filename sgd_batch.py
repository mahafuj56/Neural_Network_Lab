import numpy as np
import matplotlib.pyplot as plt

# Input
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=float)

# Target
D = np.array([0, 0, 1, 1], dtype=float)

# Learning rate
lr = 0.1

# Number of epochs
epochs = 20

# Same random result every time
np.random.seed(42)


# =========================
# SGD Delta Rule
# =========================

W_sgd = np.random.randn(3)

sgd_errors = []

for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):

        # Calculate output
        y = np.dot(W_sgd, X[i])

        # Calculate error
        e = D[i] - y

        # SGD weight update
        W_sgd += lr * e * X[i]

        # Calculate squared error
        total_error += e ** 2

    sgd_errors.append(0.5 * total_error)


# =========================
# Batch Delta Rule
# =========================

W_batch = np.random.randn(3)

batch_errors = []

for epoch in range(epochs):

    # Calculate output for all inputs
    y = X @ W_batch

    # Calculate error
    e = D - y

    # Batch weight update
    W_batch += lr * (X.T @ e) / len(X)

    # Calculate error
    batch_errors.append(0.5 * np.sum(e ** 2))


# =========================
# Display Final Weights
# =========================

print("Final SGD Weights:")
print(W_sgd)

print("\nFinal Batch Weights:")
print(W_batch)


# =========================
# Plot
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    sgd_errors,
    marker='o',
    label='SGD'
)

plt.plot(
    range(1, epochs + 1),
    batch_errors,
    marker='s',
    label='Batch'
)

plt.title("SGD vs Batch — Delta Rule")
plt.xlabel("Epoch")
plt.ylabel("Error")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()