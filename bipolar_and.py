import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Bipolar AND Input and Target
# ==========================================

X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

T = np.array([-1, -1, -1, 1])

print("Inputs:")
print(X)

print("\nTargets:")
print(T)


# ==========================================
# Learning Parameters
# ==========================================

lr = 0.1

# Initial weights
W = np.zeros(2)

# Initial bias
b = 0.0

# Maximum epochs
max_epochs = 100

print("\nInitial Weights:", W)
print("Initial Bias:", b)


# ==========================================
# Bipolar Step Function
# ==========================================

def bipolar_step(net):

    if net >= 0:
        return 1
    else:
        return -1


# ==========================================
# Perceptron Training
# ==========================================

epoch_errors = []

for epoch in range(max_epochs):

    total_error = 0

    for i in range(len(X)):

        # Calculate net input
        net = np.dot(X[i], W) + b

        # Calculate output
        y = bipolar_step(net)

        # Calculate error
        error = T[i] - y

        # Update weights
        W = W + lr * error * X[i]

        # Update bias
        b = b + lr * error

        # Total error
        total_error += abs(error)

    epoch_errors.append(total_error)

    # Check convergence
    if total_error == 0:
        print(f"\nConverged at Epoch: {epoch + 1}")
        break


# ==========================================
# Final Weights and Bias
# ==========================================

print("\nFinal Weights:", W)
print("Final Bias:", b)


# ==========================================
# Convergence Curve
# ==========================================

plt.figure(figsize=(7, 4))

plt.plot(
    range(1, len(epoch_errors) + 1),
    epoch_errors,
    marker='o',
    linewidth=2
)

plt.title("Convergence Curve (Total Error vs Epoch)")
plt.xlabel("Epoch")
plt.ylabel("Total Error")

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# Decision Boundary
# ==========================================

plt.figure(figsize=(6, 6))

# Plot data points
for i in range(len(X)):

    if T[i] == 1:
        color = 'green'
        marker = '^'
    else:
        color = 'red'
        marker = 'o'

    plt.scatter(
        X[i][0],
        X[i][1],
        color=color,
        marker=marker,
        s=150,
        zorder=5
    )

    plt.annotate(
        f"({X[i][0]},{X[i][1]}) → {T[i]}",
        (X[i][0] + 0.05, X[i][1] + 0.05)
    )


# Decision Boundary
# W[0]*x1 + W[1]*x2 + b = 0

x1_vals = np.linspace(-2, 2, 100)

if W[1] != 0:

    x2_vals = -(W[0] * x1_vals + b) / W[1]

    plt.plot(
        x1_vals,
        x2_vals,
        'k-',
        linewidth=2,
        label="Decision Boundary"
    )


plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.axhline(
    0,
    color='gray',
    linestyle='--',
    linewidth=0.8
)

plt.axvline(
    0,
    color='gray',
    linestyle='--',
    linewidth=0.8
)

plt.title("Decision Boundary — Bipolar AND")
plt.xlabel("x1")
plt.ylabel("x2")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# Testing
# ==========================================

print("\n" + "=" * 40)

print(
    f"{'x1':>5} "
    f"{'x2':>5} "
    f"{'Target':>10} "
    f"{'Output':>10}"
)

print("=" * 40)


for i in range(len(X)):

    # Calculate net
    net = np.dot(X[i], W) + b

    # Calculate output
    y = bipolar_step(net)

    print(
        f"{X[i][0]:>5} "
        f"{X[i][1]:>5} "
        f"{T[i]:>10} "
        f"{y:>10}"
    )

print("=" * 40)