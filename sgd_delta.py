import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Input and Target
# ==========================================

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=float)

D = np.array([0, 0, 1, 1], dtype=float)


# ==========================================
# Initial Parameters
# ==========================================

# Initial weights
W = np.zeros(3)

# Learning rate
lr = 0.1

# Maximum number of epochs
max_epochs = 100

# Store error of each epoch
errors = []


# ==========================================
# SGD Delta Learning Rule
# ==========================================

for epoch in range(max_epochs):

    total_error = 0

    for i in range(len(X)):

        # Calculate output
        y = np.dot(W, X[i])

        # Calculate error
        e = D[i] - y

        # Delta rule weight update
        W = W + lr * e * X[i]

        # Squared error
        total_error += e ** 2

    # Mean Squared Error
    mse = total_error / len(X)

    errors.append(mse)

    # Check convergence
    if total_error < 0.0001:
        print("Converged at epoch:", epoch + 1)
        break


# ==========================================
# Final Weights
# ==========================================

print("\nFinal Weights:")
print(W)


# ==========================================
# Convergence Curve
# ==========================================

plt.figure(figsize=(7, 4))

plt.plot(
    range(1, len(errors) + 1),
    errors,
    marker='o'
)

plt.title("SGD Convergence using Delta Learning Rule")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")

plt.grid(True)
plt.show()


# ==========================================
# Testing
# ==========================================

print("\nTesting Results")
print("=" * 50)

print(f"{'Input':>12} | {'Target':>6} | {'Output':>10} | {'Predicted':>9}")
print("-" * 50)

for i in range(len(X)):

    # Calculate output
    output = np.dot(W, X[i])

    # Convert continuous output into 0 or 1
    predicted = 1 if output >= 0.5 else 0

    print(
        f"{str(X[i][:2]):>12} | "
        f"{int(D[i]):>6} | "
        f"{output:>10.4f} | "
        f"{predicted:>9}"
    )